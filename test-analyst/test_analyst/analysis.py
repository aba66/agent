from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from .configuration import ConfigManager
from .models import AnalysisPlan, AnalysisResult, MetricMatch, MetricTargetPlan, WorkbookContext
from .utils import dataframe_preview, first_non_null, normalize_text, unique_preserve_order

DEFAULT_BASE_FIELD_ALIASES = {
    "stage": ["阶段", "phase", "stage"],
    "test_item": ["测试项目", "测试项", "test_item", "item", "项目"],
    "sn": ["sn", "sn号", "序列号", "serialnumber", "serial"],
    "test_node": ["测试节点", "节点", "testnode", "node", "station"],
    "remark": ["备注", "remark", "comment", "note"],
}

DISPLAY_NAMES = {
    "stage": "阶段",
    "test_item": "测试项目",
    "row_no": "编号",
    "sn": "SN号",
    "test_node": "测试节点",
    "remark": "remark",
    "source_file": "来源文件",
}

DECAY_METHODS = {
    "relative_to_first": "相对每个分组首条记录的变化率",
    "delta_from_first": "相对每个分组首条记录的差值",
    "delta_from_previous": "相邻两条记录的差值",
}

STATISTIC_DISPLAY_NAMES = {
    "min": "最小值",
    "max": "最大值",
    "mean": "平均值",
    "decay": "衰减率",
}


class DataAnalysisAgent:
    """Transform sheet data into a richer summary result."""

    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        self.config_manager = config_manager
        self.field_aliases = (config_manager.field_aliases() if config_manager else {}).get("base_fields", {})
        self.metric_aliases = (config_manager.field_aliases() if config_manager else {}).get("metric_aliases", {})

    def analyze(
        self,
        workbook: WorkbookContext,
        data_frames: dict[str, dict[str, pd.DataFrame]],
        analysis_plan: AnalysisPlan,
    ) -> AnalysisResult:
        sheet_name = analysis_plan.source_sheet or workbook.default_sheet
        if not sheet_name:
            raise ValueError("未能确定 source sheet。")

        frame, source_files = self._merge_source_sheet(data_frames, sheet_name)
        result = AnalysisResult(source_sheet=sheet_name, source_files=source_files)
        frame.columns = [str(column).strip() for column in frame.columns]
        frame = self._drop_embedded_header_rows(frame, result)

        result.base_field_mapping = self._detect_base_fields(frame)
        if analysis_plan.output_granularity == "group_summary" and self._plan_metric_groups(analysis_plan):
            result = self._analyze_metric_group_summary(frame, result, analysis_plan)
            return self._attach_recompute_reviews(frame, result, analysis_plan)
        if analysis_plan.output_granularity == "point_summary" and self._plan_metric_groups(analysis_plan):
            result = self._analyze_metric_group_point_summary(frame, result, analysis_plan)
            return self._attach_recompute_reviews(frame, result, analysis_plan)
        if analysis_plan.output_granularity == "group_summary":
            result = self._analyze_grouped_summary(frame, result, analysis_plan)
            return self._attach_recompute_reviews(frame, result, analysis_plan)

        metric_matches = self._match_plan_metrics(analysis_plan, list(frame.columns))
        result.metric_matches = metric_matches

        summary = pd.DataFrame(index=frame.index)
        summary[DISPLAY_NAMES["stage"]] = self._resolve_stage_column(frame, result.base_field_mapping, analysis_plan)
        summary[DISPLAY_NAMES["test_item"]] = self._resolve_test_item_column(frame, result.base_field_mapping)
        summary[DISPLAY_NAMES["row_no"]] = range(1, len(frame) + 1)
        summary[DISPLAY_NAMES["sn"]] = self._resolve_optional_column(frame, result.base_field_mapping.get("sn"))
        summary[DISPLAY_NAMES["test_node"]] = self._resolve_optional_column(frame, result.base_field_mapping.get("test_node"))
        summary[DISPLAY_NAMES["remark"]] = self._resolve_remark_column(frame, result.base_field_mapping, analysis_plan)
        if len(source_files) > 1:
            summary[DISPLAY_NAMES["source_file"]] = frame[DISPLAY_NAMES["source_file"]]
            result.comparison_notes.append(f"检测到多文件联合分析：{', '.join(source_files)}")

        group_columns = self._resolve_group_columns(summary, analysis_plan.group_dimensions)

        for metric_match in metric_matches:
            label_prefix = metric_match.requested_name
            raw_label = f"{label_prefix} 原始值"
            decay_label = f"{label_prefix} 衰减"

            if not metric_match.matched_column:
                summary[raw_label] = pd.NA
                summary[decay_label] = pd.NA
                result.warnings.append(f"测试项未找到：{metric_match.requested_name}")
                result.audit_records.append(
                    {"type": "metric_missing", "metric": metric_match.requested_name, "status": "warning"}
                )
                continue

            raw_series = frame[metric_match.matched_column]
            summary[raw_label] = raw_series

            numeric_series = pd.to_numeric(raw_series, errors="coerce")
            if numeric_series.notna().sum() == 0:
                summary[decay_label] = pd.NA
                warning = (
                    f"测试项 {metric_match.requested_name} 匹配到列 {metric_match.matched_column}，"
                    "但该列无法转成数值，已跳过衰减计算。"
                )
                result.warnings.append(warning)
                result.audit_records.append(
                    {
                        "type": "metric_non_numeric",
                        "metric": metric_match.requested_name,
                        "matched_column": metric_match.matched_column,
                        "status": "warning",
                    }
                )
            else:
                summary[decay_label] = self._compute_decay(
                    numeric_series=numeric_series,
                    group_frame=summary[group_columns] if group_columns else None,
                    method=analysis_plan.decay_method,
                )
                result.statistics[metric_match.requested_name] = self._build_metric_stats(
                    metric_match.matched_column,
                    numeric_series,
                )
                result.audit_records.append(
                    {
                        "type": "metric_computed",
                        "metric": metric_match.requested_name,
                        "matched_column": metric_match.matched_column,
                        "count": int(numeric_series.notna().sum()),
                        "status": "ok",
                    }
                )

        result.summary_frame = summary
        result.summary_preview = dataframe_preview(summary, limit=12)
        if not result.base_field_mapping.get("test_item"):
            result.warnings.append("未检测到“测试项目”字段，结果中已使用默认值“测试汇总”。")
        return self._attach_recompute_reviews(frame, result, analysis_plan)

    def summarize_review_target(
        self,
        workbook: WorkbookContext,
        data_frames: dict[str, dict[str, pd.DataFrame]],
        analysis_plan: AnalysisPlan,
        review_target: dict[str, Any] | ReviewTargetPlan,
    ) -> dict[str, Any]:
        target = review_target if isinstance(review_target, ReviewTargetPlan) else ReviewTargetPlan(**review_target)
        review_plan = analysis_plan.copy(deep=True)
        review_plan.review_targets = [target]
        review_result = self.analyze(workbook, data_frames, review_plan)
        if review_result.recompute_reviews:
            return review_result.recompute_reviews[0]
        return {
            "dimension": target.dimension,
            "value": target.value,
            "reason": target.reason,
            "status": "missing",
        }

    def _drop_embedded_header_rows(self, frame: pd.DataFrame, result: AnalysisResult) -> pd.DataFrame:
        if frame.empty:
            return frame

        normalized_columns = {column: normalize_text(column) for column in frame.columns}
        drop_indexes: list[Any] = []
        for index, row in frame.iterrows():
            matched = 0
            comparable = 0
            for column, value in row.items():
                normalized_value = normalize_text(value)
                if not normalized_value:
                    continue
                comparable += 1
                if normalized_value == normalized_columns.get(column, ""):
                    matched += 1
            if matched >= 2 and matched >= min(3, comparable):
                drop_indexes.append(index)

        if drop_indexes:
            result.audit_records.append(
                {
                    "type": "embedded_header_rows_removed",
                    "count": len(drop_indexes),
                    "status": "ok",
                }
            )
            return frame.drop(index=drop_indexes).reset_index(drop=True)
        return frame

    def _analyze_metric_group_summary(
        self,
        frame: pd.DataFrame,
        result: AnalysisResult,
        analysis_plan: AnalysisPlan,
    ) -> AnalysisResult:
        metric_groups = self._plan_metric_groups(analysis_plan)
        group_columns = self._resolve_aggregate_columns(result.base_field_mapping, analysis_plan.group_dimensions)

        if not group_columns:
            raise ValueError("当前请求要求分别统计，但原表中未识别到可用于聚合的维度字段。")

        summary_rows: list[dict[str, Any]] = []
        metric_matches: list[MetricMatch] = []
        group_frame = frame.groupby(group_columns, dropna=False, sort=False)
        total_group_count = 0

        for requested_name in self._plan_metric_labels(analysis_plan):
            columns = [column for column in metric_groups.get(requested_name, []) if column in frame.columns]
            metric_matches.append(
                MetricMatch(
                    requested_name=requested_name,
                    matched_column=", ".join(columns) if columns else None,
                    score=1.0 if columns else 0.0,
                    reason="metric group expansion" if columns else "group columns not found",
                    status="matched" if columns else "missing",
                )
            )
            if not columns:
                result.warnings.append(f"指标组未找到可用列：{requested_name}")

        for group_key, group_df in group_frame:
            total_group_count += 1
            summary_row = self._build_group_identity_row(group_columns, group_key, group_df, result.base_field_mapping)
            summary_row["记录数"] = int(len(group_df))
            has_metric_value = False
            requested_statistics = self._resolve_requested_statistics(analysis_plan.statistics)

            for requested_name in self._plan_metric_labels(analysis_plan):
                columns = [column for column in metric_groups.get(requested_name, []) if column in group_df.columns]
                if not columns:
                    for statistic in requested_statistics:
                        summary_row[f"{requested_name} {STATISTIC_DISPLAY_NAMES[statistic]}"] = pd.NA
                    continue

                numeric_block = group_df[columns].apply(pd.to_numeric, errors="coerce")
                stats_row, audit_record = self._summarize_metric_group(
                    requested_name=requested_name,
                    numeric_block=numeric_block,
                    method=analysis_plan.decay_method,
                    requested_statistics=requested_statistics,
                )
                if any(value is not None and not pd.isna(value) for value in stats_row.values()):
                    has_metric_value = True
                summary_row.update(stats_row)
                audit_record["group"] = {
                    self._display_name_for_source_column(column, result.base_field_mapping): value
                    for column, value in zip(group_columns, self._coerce_group_key(group_key))
                }
                result.audit_records.append(audit_record)

            if has_metric_value:
                summary_rows.append(summary_row)
            else:
                result.warnings.append(
                    "以下分组没有可用数值，已从汇总结果中跳过："
                    + ", ".join(str(value) for value in self._coerce_group_key(group_key))
                )

        result.metric_matches = metric_matches
        result.summary_frame = pd.DataFrame(summary_rows)
        result.summary_preview = dataframe_preview(result.summary_frame, limit=12)
        result.statistics = {
            "summary_mode": {
                "analysis_mode": analysis_plan.output_granularity,
                "grouped_by": [
                    self._display_name_for_source_column(column, result.base_field_mapping)
                    for column in group_columns
                ],
                "group_count": total_group_count,
                "metric_groups": {
                    key: {"column_count": len(value), "columns": value}
                    for key, value in metric_groups.items()
                },
            }
        }
        return result

    def _analyze_grouped_summary(
        self,
        frame: pd.DataFrame,
        result: AnalysisResult,
        analysis_plan: AnalysisPlan,
    ) -> AnalysisResult:
        group_columns = self._resolve_aggregate_columns(result.base_field_mapping, analysis_plan.group_dimensions)

        if not group_columns:
            raise ValueError("当前请求要求分别统计，但原表中未识别到可用于聚合的维度字段。")

        metric_matches = self._match_plan_metrics(analysis_plan, list(frame.columns))
        requested_statistics = self._resolve_requested_statistics(analysis_plan.statistics)
        summary_rows: list[dict[str, Any]] = []
        group_frame = frame.groupby(group_columns, dropna=False, sort=False)
        total_group_count = 0

        for group_key, group_df in group_frame:
            total_group_count += 1
            summary_row = self._build_group_identity_row(group_columns, group_key, group_df, result.base_field_mapping)
            summary_row["记录数"] = int(len(group_df))

            for metric_match in metric_matches:
                for statistic in requested_statistics:
                    summary_row[f"{metric_match.requested_name} {STATISTIC_DISPLAY_NAMES[statistic]}"] = pd.NA

                if not metric_match.matched_column:
                    result.warnings.append(f"测试项未找到：{metric_match.requested_name}")
                    continue

                numeric_series = pd.to_numeric(group_df[metric_match.matched_column], errors="coerce")
                stats_row = self._summarize_metric_series(
                    requested_name=metric_match.requested_name,
                    numeric_series=numeric_series,
                    method=analysis_plan.decay_method,
                    requested_statistics=requested_statistics,
                )
                summary_row.update(stats_row)
                result.audit_records.append(
                    {
                        "type": "grouped_summary",
                        "metric": metric_match.requested_name,
                        "matched_column": metric_match.matched_column,
                        "group": {
                            self._display_name_for_source_column(column, result.base_field_mapping): value
                            for column, value in zip(group_columns, self._coerce_group_key(group_key))
                        },
                        "value_count": int(numeric_series.dropna().count()),
                        "requested_statistics": requested_statistics,
                        "decay_method": analysis_plan.decay_method,
                        "status": "ok" if not numeric_series.dropna().empty else "warning",
                    }
                )

            summary_rows.append(summary_row)

        result.metric_matches = metric_matches
        result.summary_frame = pd.DataFrame(summary_rows)
        result.summary_preview = dataframe_preview(result.summary_frame, limit=12)
        result.statistics = {
            "summary_mode": {
                "analysis_mode": analysis_plan.output_granularity,
                "grouped_by": [
                    self._display_name_for_source_column(column, result.base_field_mapping)
                    for column in group_columns
                ],
                "group_count": total_group_count,
                "requested_statistics": requested_statistics,
            }
        }
        return result

    def _analyze_metric_group_point_summary(
        self,
        frame: pd.DataFrame,
        result: AnalysisResult,
        analysis_plan: AnalysisPlan,
    ) -> AnalysisResult:
        metric_groups = self._plan_metric_groups(analysis_plan)
        group_columns = self._resolve_aggregate_columns(result.base_field_mapping, analysis_plan.group_dimensions)

        if not group_columns:
            raise ValueError("当前请求要求按点位分别统计，但原表中未识别到可用于聚合的维度字段。")

        requested_statistics = self._resolve_requested_statistics(analysis_plan.statistics)
        summary_rows: list[dict[str, Any]] = []
        metric_matches: list[MetricMatch] = []
        group_frame = frame.groupby(group_columns, dropna=False, sort=False)
        total_group_count = 0

        for requested_name in self._plan_metric_labels(analysis_plan):
            columns = [column for column in metric_groups.get(requested_name, []) if column in frame.columns]
            metric_matches.append(
                MetricMatch(
                    requested_name=requested_name,
                    matched_column=", ".join(columns) if columns else None,
                    score=1.0 if columns else 0.0,
                    reason="point-level metric group expansion" if columns else "group columns not found",
                    status="matched" if columns else "missing",
                )
            )
            if not columns:
                result.warnings.append(f"指标组未找到可用列：{requested_name}")

        for group_key, group_df in group_frame:
            total_group_count += 1
            summary_row = self._build_group_identity_row(group_columns, group_key, group_df, result.base_field_mapping)
            summary_row["记录数"] = int(len(group_df))

            for requested_name in self._plan_metric_labels(analysis_plan):
                columns = [column for column in metric_groups.get(requested_name, []) if column in group_df.columns]
                for column in columns:
                    numeric_series = pd.to_numeric(group_df[column], errors="coerce")
                    stats_row = self._summarize_metric_series(
                        requested_name=column,
                        numeric_series=numeric_series,
                        method=analysis_plan.decay_method,
                        requested_statistics=requested_statistics,
                    )
                    summary_row.update(stats_row)
                    result.audit_records.append(
                        {
                            "type": "metric_group_point_summary",
                            "metric_family": requested_name,
                            "metric": column,
                            "group": {
                                self._display_name_for_source_column(group_column, result.base_field_mapping): value
                                for group_column, value in zip(group_columns, self._coerce_group_key(group_key))
                            },
                            "value_count": int(numeric_series.dropna().count()),
                            "requested_statistics": requested_statistics,
                            "decay_method": analysis_plan.decay_method,
                            "status": "ok" if not numeric_series.dropna().empty else "warning",
                        }
                    )

            summary_rows.append(summary_row)

        result.metric_matches = metric_matches
        result.summary_frame = pd.DataFrame(summary_rows)
        result.summary_preview = dataframe_preview(result.summary_frame, limit=12)
        result.statistics = {
            "summary_mode": {
                "analysis_mode": analysis_plan.output_granularity,
                "metric_group_mode": "point_summary",
                "grouped_by": [
                    self._display_name_for_source_column(column, result.base_field_mapping)
                    for column in group_columns
                ],
                "group_count": total_group_count,
                "requested_statistics": requested_statistics,
                "metric_groups": {
                    key: {"column_count": len(value), "columns": value}
                    for key, value in metric_groups.items()
                },
            }
        }
        return result

    def _merge_source_sheet(
        self,
        data_frames: dict[str, dict[str, pd.DataFrame]],
        sheet_name: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        collected_frames: list[pd.DataFrame] = []
        source_files: list[str] = []
        for file_id, sheet_map in data_frames.items():
            if sheet_name not in sheet_map:
                continue
            file_name = self._build_source_label(file_id, source_files)
            frame = sheet_map[sheet_name].copy()
            frame[DISPLAY_NAMES["source_file"]] = file_name
            collected_frames.append(frame)
            source_files.append(file_name)

        if not collected_frames:
            raise ValueError(f"在所有输入文件中都没有找到 source sheet：{sheet_name}")

        merged = pd.concat(collected_frames, ignore_index=True, sort=False)
        return merged, source_files

    def _build_source_label(self, file_id: str, existing_labels: list[str]) -> str:
        path = Path(file_id)
        candidate = path.name
        if candidate not in existing_labels:
            return candidate
        if path.parent.name:
            candidate = f"{path.parent.name}/{path.name}"
        if candidate not in existing_labels:
            return candidate
        return str(path)

    def _detect_base_fields(
        self,
        frame: pd.DataFrame,
    ) -> dict[str, str]:
        aliases = {**DEFAULT_BASE_FIELD_ALIASES, **self.field_aliases}
        mapping: dict[str, str] = {}
        normalized_columns = {normalize_text(column): column for column in frame.columns}

        for canonical, alias_list in aliases.items():
            for alias in alias_list:
                matched = normalized_columns.get(normalize_text(alias))
                if matched:
                    mapping[canonical] = matched
                    break
        return mapping

    def _match_plan_metrics(self, analysis_plan: AnalysisPlan, columns: list[str]) -> list[MetricMatch]:
        requested_metrics = self._plan_metric_labels(analysis_plan)
        normalized_columns = {column: normalize_text(column) for column in columns}
        matches: list[MetricMatch] = []
        target_lookup = {target.label: target for target in analysis_plan.metric_targets}

        for requested in unique_preserve_order(requested_metrics):
            target = target_lookup.get(requested)
            if target and target.target_type == "metric_group":
                matches.append(
                    MetricMatch(
                        requested_name=requested,
                        matched_column=", ".join(target.columns) if target.columns else None,
                        score=1.0 if target.columns else 0.0,
                        reason="metric group expansion" if target.columns else "group columns not found",
                        status="matched" if target.columns else "missing",
                    )
                )
                continue
            if target and target.matched_column in columns:
                matches.append(
                    MetricMatch(
                        requested_name=requested,
                        matched_column=target.matched_column,
                        score=1.0,
                        reason="analysis plan resolved match",
                        status="matched",
                    )
                )
                continue
            candidates = [requested] + self.metric_aliases.get(requested, [])
            requested_match = None
            for candidate in candidates:
                requested_match = self._match_single_metric(candidate, columns, normalized_columns, requested)
                if requested_match and requested_match.matched_column:
                    break
            if requested_match is None:
                requested_match = MetricMatch(requested_name=requested, reason="not found", status="missing")
            matches.append(requested_match)

        return matches

    def _match_single_metric(
        self,
        candidate_name: str,
        columns: list[str],
        normalized_columns: dict[str, str],
        requested_name: str,
    ) -> MetricMatch:
        candidate_norm = normalize_text(candidate_name)
        exact = next((column for column, normalized in normalized_columns.items() if normalized == candidate_norm), None)
        if exact:
            return MetricMatch(
                requested_name=requested_name,
                matched_column=exact,
                score=1.0,
                reason="exact match",
                status="matched",
            )

        contains = next(
            (
                column
                for column, normalized in normalized_columns.items()
                if candidate_norm and (candidate_norm in normalized or normalized in candidate_norm)
            ),
            None,
        )
        if contains:
            return MetricMatch(
                requested_name=requested_name,
                matched_column=contains,
                score=0.85,
                reason=f"contains match via {candidate_name}",
                status="matched",
            )

        best_column = None
        best_score = 0.0
        for column, normalized in normalized_columns.items():
            score = SequenceMatcher(a=candidate_norm, b=normalized).ratio()
            if score > best_score:
                best_score = score
                best_column = column

        if best_column and best_score >= 0.65:
            return MetricMatch(
                requested_name=requested_name,
                matched_column=best_column,
                score=round(best_score, 3),
                reason=f"fuzzy match via {candidate_name}",
                status="matched",
            )
        return MetricMatch(requested_name=requested_name, score=best_score, reason="not found", status="missing")

    def _resolve_stage_column(
        self,
        frame: pd.DataFrame,
        mapping: dict[str, str],
        analysis_plan: AnalysisPlan,
    ) -> pd.Series:
        if analysis_plan.stage_value:
            return pd.Series([analysis_plan.stage_value] * len(frame), index=frame.index)
        if mapping.get("stage"):
            return frame[mapping["stage"]]
        return pd.Series([""] * len(frame), index=frame.index)

    def _resolve_test_item_column(self, frame: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
        if mapping.get("test_item"):
            return frame[mapping["test_item"]]
        return pd.Series(["测试汇总"] * len(frame), index=frame.index)

    def _resolve_optional_column(self, frame: pd.DataFrame, column: str | None) -> pd.Series:
        if column:
            return frame[column]
        return pd.Series([""] * len(frame), index=frame.index)

    def _resolve_remark_column(
        self,
        frame: pd.DataFrame,
        mapping: dict[str, str],
        analysis_plan: AnalysisPlan,
    ) -> pd.Series:
        if analysis_plan.remark_value:
            return pd.Series([analysis_plan.remark_value] * len(frame), index=frame.index)
        if mapping.get("remark"):
            return frame[mapping["remark"]]
        return pd.Series([""] * len(frame), index=frame.index)

    def _resolve_group_columns(self, summary: pd.DataFrame, requested_group_by: list[str]) -> list[str]:
        group_columns: list[str] = []
        requested = set(self._normalize_base_field_names(requested_group_by))
        if "stage" in requested and DISPLAY_NAMES["stage"] in summary.columns:
            group_columns.append(DISPLAY_NAMES["stage"])
        if "test_item" in requested and DISPLAY_NAMES["test_item"] in summary.columns:
            group_columns.append(DISPLAY_NAMES["test_item"])
        if "sn" in requested and DISPLAY_NAMES["sn"] in summary.columns:
            group_columns.append(DISPLAY_NAMES["sn"])
        if "test_node" in requested and DISPLAY_NAMES["test_node"] in summary.columns:
            group_columns.append(DISPLAY_NAMES["test_node"])
        if "source_file" in requested and DISPLAY_NAMES["source_file"] in summary.columns:
            group_columns.append(DISPLAY_NAMES["source_file"])
        return group_columns

    def _resolve_aggregate_columns(
        self,
        base_field_mapping: dict[str, str],
        aggregate_by_fields: list[str],
    ) -> list[str]:
        columns: list[str] = []
        for field in self._normalize_base_field_names(aggregate_by_fields):
            mapped = base_field_mapping.get(field)
            if mapped:
                columns.append(mapped)
        return unique_preserve_order(columns)

    def _normalize_base_field_names(self, field_names: list[str]) -> list[str]:
        alias_lookup: dict[str, str] = {}
        aliases = {**DEFAULT_BASE_FIELD_ALIASES, **self.field_aliases}
        for canonical, alias_list in aliases.items():
            alias_lookup[normalize_text(canonical)] = canonical
            alias_lookup[normalize_text(DISPLAY_NAMES.get(canonical, canonical))] = canonical
            for alias in alias_list:
                alias_lookup[normalize_text(alias)] = canonical

        normalized: list[str] = []
        for field_name in field_names:
            text = str(field_name).strip()
            if not text:
                continue
            normalized.append(alias_lookup.get(normalize_text(text), text))
        return unique_preserve_order(normalized)

    def _build_group_identity_row(
        self,
        group_columns: list[str],
        group_key: Any,
        group_df: pd.DataFrame,
        base_field_mapping: dict[str, str],
    ) -> dict[str, Any]:
        row: dict[str, Any] = {}
        key_values = self._coerce_group_key(group_key)
        for column, value in zip(group_columns, key_values):
            row[self._display_name_for_source_column(column, base_field_mapping)] = value

        for canonical in ("stage", "test_item"):
            source_column = base_field_mapping.get(canonical)
            display_name = DISPLAY_NAMES.get(canonical)
            if not source_column or not display_name or display_name in row:
                continue
            if source_column in group_df.columns:
                distinct_values = group_df[source_column].dropna().astype(str).str.strip()
                distinct_values = distinct_values[distinct_values.ne("")]
                if distinct_values.nunique() == 1:
                    row[display_name] = distinct_values.iloc[0]
        return row

    def _coerce_group_key(self, group_key: Any) -> tuple[Any, ...]:
        if isinstance(group_key, tuple):
            return group_key
        return (group_key,)

    def _display_name_for_source_column(
        self,
        source_column: str,
        base_field_mapping: dict[str, str],
    ) -> str:
        for canonical, mapped in base_field_mapping.items():
            if mapped == source_column:
                return DISPLAY_NAMES.get(canonical, source_column)
        return source_column

    def _summarize_metric_group(
        self,
        requested_name: str,
        numeric_block: pd.DataFrame,
        method: str | None,
        requested_statistics: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        flattened = pd.Series(numeric_block.to_numpy().ravel()).dropna()
        family_mean_series = numeric_block.mean(axis=1, skipna=True)
        family_mean_series = pd.to_numeric(family_mean_series, errors="coerce")
        decay_series = self._compute_decay(family_mean_series, group_frame=None, method=method)
        decay_value = self._last_non_null(decay_series)

        stats_row: dict[str, Any] = {}
        if "min" in requested_statistics:
            stats_row[f"{requested_name} 最小值"] = None if flattened.empty else round(float(flattened.min()), 6)
        if "max" in requested_statistics:
            stats_row[f"{requested_name} 最大值"] = None if flattened.empty else round(float(flattened.max()), 6)
        if "mean" in requested_statistics:
            stats_row[f"{requested_name} 平均值"] = None if flattened.empty else round(float(flattened.mean()), 6)
        if "decay" in requested_statistics:
            stats_row[f"{requested_name} 衰减率"] = None if pd.isna(decay_value) else round(float(decay_value), 6)
        audit_record = {
            "type": "metric_group_summary",
            "metric": requested_name,
            "column_count": int(numeric_block.shape[1]),
            "value_count": int(flattened.count()),
            "decay_method": method,
            "status": "ok" if not flattened.empty else "warning",
        }
        return stats_row, audit_record

    def _summarize_metric_series(
        self,
        requested_name: str,
        numeric_series: pd.Series,
        method: str | None,
        requested_statistics: list[str],
    ) -> dict[str, Any]:
        valid = numeric_series.dropna()
        decay_series = self._compute_decay(valid, group_frame=None, method=method)
        decay_value = self._last_non_null(decay_series)

        stats_row: dict[str, Any] = {}
        if "min" in requested_statistics:
            stats_row[f"{requested_name} 最小值"] = None if valid.empty else round(float(valid.min()), 6)
        if "max" in requested_statistics:
            stats_row[f"{requested_name} 最大值"] = None if valid.empty else round(float(valid.max()), 6)
        if "mean" in requested_statistics:
            stats_row[f"{requested_name} 平均值"] = None if valid.empty else round(float(valid.mean()), 6)
        if "decay" in requested_statistics:
            stats_row[f"{requested_name} 衰减率"] = None if pd.isna(decay_value) else round(float(decay_value), 6)
        return stats_row

    def _resolve_requested_statistics(self, requested_statistics: list[str]) -> list[str]:
        supported = ["min", "max", "mean", "decay"]
        return [item for item in requested_statistics if item in supported]

    def _plan_metric_groups(self, analysis_plan: AnalysisPlan) -> dict[str, list[str]]:
        return {
            target.label: list(target.columns)
            for target in analysis_plan.metric_targets
            if target.target_type == "metric_group" and target.columns
        }

    def _plan_metric_labels(self, analysis_plan: AnalysisPlan) -> list[str]:
        return [target.label for target in analysis_plan.metric_targets if target.label]

    def _attach_recompute_reviews(
        self,
        frame: pd.DataFrame,
        result: AnalysisResult,
        analysis_plan: AnalysisPlan,
    ) -> AnalysisResult:
        if not analysis_plan.review_targets:
            return result

        metric_groups = self._plan_metric_groups(analysis_plan)
        reviews: list[dict[str, Any]] = []
        for review_target in analysis_plan.review_targets:
            mapped_column = result.base_field_mapping.get(review_target.dimension)
            if not mapped_column or mapped_column not in frame.columns:
                continue
            filtered = frame[frame[mapped_column].astype(str) == review_target.value].copy()
            if filtered.empty:
                reviews.append(
                    {
                        "dimension": review_target.dimension,
                        "value": review_target.value,
                        "reason": review_target.reason,
                        "status": "missing",
                    }
                )
                continue

            group_columns = self._resolve_aggregate_columns(result.base_field_mapping, analysis_plan.group_dimensions)
            recomputed_rows: list[dict[str, Any]] = []
            grouped = filtered.groupby(group_columns, dropna=False, sort=False) if group_columns else [(None, filtered)]
            requested_labels = set(review_target.metric_labels or self._plan_metric_labels(analysis_plan))
            for group_key, group_df in grouped:
                row = self._build_group_identity_row(group_columns, group_key, group_df, result.base_field_mapping) if group_columns else {}
                row["记录数"] = int(len(group_df))
                resolved_group = {
                    self._display_name_for_source_column(group_column, result.base_field_mapping): value
                    for group_column, value in zip(group_columns, self._coerce_group_key(group_key))
                }
                row["原始分组键"] = resolved_group
                row["过滤条件"] = {"dimension": review_target.dimension, "value": review_target.value}
                row["原始记录预览"] = group_df.head(20).to_dict(orient="records")
                row["复算指标"] = []
                row["聚合口径"] = {
                    "group_dimensions": list(analysis_plan.group_dimensions),
                    "statistics": self._resolve_requested_statistics(analysis_plan.statistics),
                    "decay_method": analysis_plan.decay_method,
                    "output_granularity": analysis_plan.output_granularity,
                }
                for label, columns in metric_groups.items():
                    if requested_labels and label not in requested_labels:
                        continue
                    usable_columns = [column for column in columns if column in group_df.columns]
                    if not usable_columns:
                        continue
                    numeric_block = group_df[usable_columns].apply(pd.to_numeric, errors="coerce")
                    stats_row, _ = self._summarize_metric_group(
                        requested_name=label,
                        numeric_block=numeric_block,
                        method=analysis_plan.decay_method,
                        requested_statistics=self._resolve_requested_statistics(analysis_plan.statistics),
                    )
                    row.update(stats_row)
                    flattened = pd.Series(numeric_block.to_numpy().ravel()).dropna()
                    family_mean_series = pd.to_numeric(numeric_block.mean(axis=1, skipna=True), errors="coerce")
                    decay_series = self._compute_decay(
                        family_mean_series,
                        group_frame=None,
                        method=analysis_plan.decay_method,
                    )
                    row["复算指标"].append(
                        {
                            "label": label,
                            "columns": usable_columns,
                            "raw_values_by_column": {
                                column: [
                                    None if pd.isna(value) else float(value)
                                    for value in pd.to_numeric(group_df[column], errors="coerce").tolist()
                                ]
                                for column in usable_columns
                            },
                            "flattened_values": [
                                None if pd.isna(value) else float(value)
                                for value in flattened.tolist()
                            ],
                            "row_mean_values": [
                                None if pd.isna(value) else float(value)
                                for value in family_mean_series.tolist()
                            ],
                            "decay_series": [
                                None if pd.isna(value) else float(value)
                                for value in pd.to_numeric(decay_series, errors="coerce").tolist()
                            ],
                            "statistics": stats_row,
                        }
                    )
                for target in analysis_plan.metric_targets:
                    if target.target_type != "metric_series" or target.label not in requested_labels:
                        continue
                    matched_column = target.matched_column or target.label
                    if matched_column not in group_df.columns:
                        continue
                    numeric_series = pd.to_numeric(group_df[matched_column], errors="coerce")
                    stats_row = self._summarize_metric_series(
                        requested_name=target.label,
                        numeric_series=numeric_series,
                        method=analysis_plan.decay_method,
                        requested_statistics=self._resolve_requested_statistics(analysis_plan.statistics),
                    )
                    row.update(stats_row)
                    row["复算指标"].append(
                        {
                            "label": target.label,
                            "columns": [matched_column],
                            "raw_values_by_column": {
                                matched_column: [
                                    None if pd.isna(value) else float(value)
                                    for value in numeric_series.tolist()
                                ]
                            },
                            "flattened_values": [
                                None if pd.isna(value) else float(value)
                                for value in numeric_series.dropna().tolist()
                            ],
                            "row_mean_values": [
                                None if pd.isna(value) else float(value)
                                for value in numeric_series.tolist()
                            ],
                            "decay_series": [
                                None if pd.isna(value) else float(value)
                                for value in pd.to_numeric(
                                    self._compute_decay(numeric_series, group_frame=None, method=analysis_plan.decay_method),
                                    errors="coerce",
                                ).tolist()
                            ],
                            "statistics": stats_row,
                        }
                    )
                row["结果回放"] = self._locate_summary_row_replay(
                    summary_frame=result.summary_frame,
                    grouped_identity=resolved_group,
                    computed_row=row,
                    review_filter=row["过滤条件"],
                )
                recomputed_rows.append(row)

            reviews.append(
                {
                    "dimension": review_target.dimension,
                    "value": review_target.value,
                    "reason": review_target.reason,
                    "status": "ok",
                    "matched_row_count": int(len(filtered)),
                    "matched_columns": list(filtered.columns),
                    "rows": recomputed_rows,
                }
            )
        result.recompute_reviews = reviews
        return result

    def _locate_summary_row_replay(
        self,
        *,
        summary_frame: pd.DataFrame | None,
        grouped_identity: dict[str, Any],
        computed_row: dict[str, Any],
        review_filter: dict[str, Any],
    ) -> dict[str, Any]:
        if summary_frame is None or summary_frame.empty:
            return {"status": "missing_summary"}

        if not grouped_identity:
            preview = summary_frame.head(3).to_dict(orient="records")
            return {
                "status": "summary_without_group_key",
                "candidate_preview": preview,
            }

        filter_dimension = str(review_filter.get("dimension", "")).strip()
        if filter_dimension and filter_dimension not in grouped_identity:
            matched = summary_frame.copy()
            for column, value in grouped_identity.items():
                if column not in matched.columns:
                    return {
                        "status": "filtered_subset_only",
                        "reason": "当前复核对象是汇总分组下的子集，系统已回放原始值和聚合过程，但无法直接逐值对比汇总行。",
                    }
                matched = matched[matched[column].astype(str) == str(value)]
            return {
                "status": "filtered_subset_only",
                "reason": "当前复核对象是汇总分组下的子集，系统已回放原始值和聚合过程，但无法直接逐值对比汇总行。",
                "candidate_summary_rows": matched.head(3).to_dict(orient="records"),
            }

        matched = summary_frame.copy()
        for column, value in grouped_identity.items():
            if column not in matched.columns:
                return {
                    "status": "summary_group_column_missing",
                    "column": column,
                }
            matched = matched[matched[column].astype(str) == str(value)]

        if matched.empty:
            return {
                "status": "summary_row_not_found",
                "grouped_identity": grouped_identity,
            }

        summary_row = matched.iloc[0].to_dict()
        compared_metrics: list[dict[str, Any]] = []
        for column, value in computed_row.items():
            if column not in summary_row or column in {"原始分组键", "过滤条件", "原始记录预览", "复算指标", "聚合口径", "结果回放"}:
                continue
            if isinstance(value, (dict, list)):
                continue
            summary_value = summary_row.get(column)
            compared_metrics.append(
                {
                    "column": column,
                    "summary_value": summary_value,
                    "recomputed_value": value,
                    "matches": self._values_match(summary_value, value),
                }
            )
        return {
            "status": "ok",
            "summary_row": summary_row,
            "comparisons": compared_metrics,
        }

    def _values_match(self, left: Any, right: Any) -> bool:
        if pd.isna(left) and pd.isna(right):
            return True
        try:
            return round(float(left), 6) == round(float(right), 6)
        except Exception:
            return str(left) == str(right)

    def _last_non_null(self, series: pd.Series) -> Any:
        valid = pd.to_numeric(series, errors="coerce").dropna()
        if valid.empty:
            return pd.NA
        return valid.iloc[-1]

    def _compute_decay(
        self,
        numeric_series: pd.Series,
        group_frame: pd.DataFrame | None,
        method: str | None,
    ) -> pd.Series:
        if not method:
            return pd.Series([pd.NA] * len(numeric_series), index=numeric_series.index)

        if group_frame is not None and not group_frame.empty:
            temp = group_frame.copy()
            temp["__value__"] = numeric_series
            grouped = temp.groupby(list(group_frame.columns), dropna=False)["__value__"]
            baseline = grouped.transform(first_non_null)
            previous = grouped.shift(1)
        else:
            baseline_value = first_non_null(numeric_series)
            baseline = pd.Series([baseline_value] * len(numeric_series), index=numeric_series.index)
            previous = numeric_series.shift(1)

        if method == "relative_to_first":
            denominator = baseline.replace({0: pd.NA})
            result = (numeric_series - baseline) / denominator
            return pd.to_numeric(result, errors="coerce").round(6)
        if method == "delta_from_first":
            result = numeric_series - baseline
            return pd.to_numeric(result, errors="coerce").round(6)
        if method == "delta_from_previous":
            result = numeric_series - previous
            return pd.to_numeric(result, errors="coerce").round(6)
        return pd.Series([pd.NA] * len(numeric_series), index=numeric_series.index)

    def _build_metric_stats(self, matched_column: str, numeric_series: pd.Series) -> dict[str, float | int | str | None]:
        valid = numeric_series.dropna()
        return {
            "matched_column": matched_column,
            "count": int(valid.count()),
            "mean": None if valid.empty else round(float(valid.mean()), 6),
            "min": None if valid.empty else round(float(valid.min()), 6),
            "max": None if valid.empty else round(float(valid.max()), 6),
            "std": None if len(valid) <= 1 else round(float(valid.std()), 6),
        }


class ValidationAgent:
    """Run lightweight validations on the generated result."""

    def validate(self, analysis_plan: AnalysisPlan, analysis_result: AnalysisResult) -> list[str]:
        messages: list[str] = []
        missing = [match.requested_name for match in analysis_result.metric_matches if not match.matched_column]
        if missing:
            messages.append(f"以下测试项未匹配到原始列：{', '.join(missing)}")

        if analysis_result.summary_frame is None or analysis_result.summary_frame.empty:
            messages.append("汇总结果为空，请检查输入文件和筛选条件。")
            return messages

        required_columns = ["阶段", "测试项目", "编号", "SN号", "测试节点", "remark"]
        if analysis_plan.output_granularity in {"group_summary", "point_summary"}:
            required_columns = []
        missing_columns = [column for column in required_columns if column not in analysis_result.summary_frame.columns]
        if missing_columns:
            messages.append(f"输出结果缺少基础列：{', '.join(missing_columns)}")

        expected_pairs = []
        if analysis_plan.output_granularity in {"group_summary", "point_summary"}:
            requested_statistics = [item for item in analysis_plan.statistics if item in STATISTIC_DISPLAY_NAMES]
            metric_groups = {
                target.label: list(target.columns)
                for target in analysis_plan.metric_targets
                if target.target_type == "metric_group"
            }
            if analysis_plan.output_granularity == "point_summary":
                for columns in metric_groups.values():
                    for column in columns:
                        expected_pairs.extend([f"{column} {STATISTIC_DISPLAY_NAMES[item]}" for item in requested_statistics])
            else:
                for target in analysis_plan.metric_targets:
                    if target.label:
                        expected_pairs.extend([f"{target.label} {STATISTIC_DISPLAY_NAMES[item]}" for item in requested_statistics])
        else:
            for target in analysis_plan.metric_targets:
                if target.label:
                    expected_pairs.extend([f"{target.label} 原始值", f"{target.label} 衰减"])
        absent_pairs = [column for column in expected_pairs if column not in analysis_result.summary_frame.columns]
        if absent_pairs:
            messages.append(f"输出结果缺少指标列：{', '.join(absent_pairs)}")

        if analysis_plan.threshold_rules:
            messages.append("检测到阈值规则输入，当前版本已记录规则，建议人工复核判定结论。")

        if analysis_plan.review_targets:
            if analysis_result.recompute_reviews:
                mismatch_targets = []
                for review in analysis_result.recompute_reviews:
                    for row in review.get("rows", []):
                        replay = row.get("结果回放", {})
                        comparisons = replay.get("comparisons", [])
                        status = replay.get("status")
                        if status in {"filtered_subset_only", "summary_without_group_key"}:
                            continue
                        if status != "ok" or any(not item.get("matches", False) for item in comparisons):
                            mismatch_targets.append(f"{review.get('dimension')}={review.get('value')}")
                            break
                if mismatch_targets:
                    messages.append("可复算复核已执行，但以下对象的回放结果与汇总表未完全对齐：" + ", ".join(mismatch_targets))
                else:
                    messages.append("已生成可复算复核结果，并回放了指定分组的原始值、聚合过程与统计结果。")
            else:
                messages.append("用户请求了结果复核，但当前没有生成可复算复核结果。")

        if not messages:
            messages.append("校验通过：基础列、指标列和结果数据已生成。")
        return messages
