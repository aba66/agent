from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from .models import InputFileContext, SheetSummary, WorkbookContext
from .utils import dataframe_preview, detect_header_candidates


class DataParserAgent:
    """Load one or more workbooks and summarize their structure."""

    def load(self, file_path: str | Path, role: str = "data") -> tuple[InputFileContext, dict[str, pd.DataFrame]]:
        path = Path(file_path).expanduser().resolve()
        suffix = path.suffix.lower()
        file_warnings: list[str] = []

        if suffix == ".csv":
            frame = pd.read_csv(path)
            raw_preview = pd.read_csv(path, header=None, nrows=8)
            sheet_name = "Sheet1"
            context = InputFileContext(
                file_path=path,
                file_type="csv",
                role=role,
                default_sheet=sheet_name,
                warnings=file_warnings,
                sheet_summaries=[
                    SheetSummary(
                        name=sheet_name,
                        rows=int(frame.shape[0]),
                        columns=int(frame.shape[1]),
                        column_names=[str(column) for column in frame.columns],
                        header_candidates=detect_header_candidates(raw_preview),
                        preview_records=dataframe_preview(frame),
                    )
                ],
            )
            return context, {sheet_name: frame}

        if suffix not in {".xlsx", ".xls", ".xlsm"}:
            raise ValueError(f"暂不支持的文件类型：{suffix}")

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="wmf image format is not supported.*", category=UserWarning)
                warnings.filterwarnings("ignore", message="Unable to read chart.*", category=UserWarning)
                excel_file = pd.ExcelFile(path)
        except ImportError as exc:
            raise ValueError(
                "读取 Excel 失败。请确认已安装所需依赖；若是 .xls 文件，通常还需要 xlrd。"
            ) from exc

        frames: dict[str, pd.DataFrame] = {}
        summaries: list[SheetSummary] = []
        for sheet_name in excel_file.sheet_names:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="wmf image format is not supported.*", category=UserWarning)
                warnings.filterwarnings("ignore", message="Unable to read chart.*", category=UserWarning)
                frame = pd.read_excel(path, sheet_name=sheet_name)
                raw_preview = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=8)
            frames[sheet_name] = frame
            summaries.append(
                SheetSummary(
                    name=sheet_name,
                    rows=int(frame.shape[0]),
                    columns=int(frame.shape[1]),
                    column_names=[str(column) for column in frame.columns],
                    header_candidates=detect_header_candidates(raw_preview),
                    preview_records=dataframe_preview(frame),
                )
            )

        default_sheet = summaries[0].name if summaries else None
        context = InputFileContext(
            file_path=path,
            file_type="excel",
            role=role,
            default_sheet=default_sheet,
            warnings=file_warnings,
            sheet_summaries=summaries,
        )
        return context, frames

    def load_many(
        self,
        data_files: list[str | Path],
        template_file: str | Path | None = None,
    ) -> tuple[WorkbookContext, dict[str, dict[str, pd.DataFrame]], dict[str, pd.DataFrame] | None]:
        input_contexts: list[InputFileContext] = []
        data_frames: dict[str, dict[str, pd.DataFrame]] = {}
        aggregated_summaries: list[SheetSummary] = []
        all_warnings: list[str] = []

        for file_path in data_files:
            file_context, frames = self.load(file_path, role="data")
            input_contexts.append(file_context)
            data_frames[str(file_context.file_path)] = frames
            all_warnings.extend(file_context.warnings)
            for summary in file_context.sheet_summaries:
                if any(item.name == summary.name for item in aggregated_summaries):
                    continue
                aggregated_summaries.append(summary)

        template_context = None
        template_frames = None
        if template_file:
            template_context, template_frames = self.load(template_file, role="template")
            input_contexts.append(template_context)
            all_warnings.extend(template_context.warnings)

        primary = next((item for item in input_contexts if item.role == "data"), None)
        workbook = WorkbookContext(
            file_path=primary.file_path if primary else None,
            file_type="multi",
            sheet_summaries=aggregated_summaries,
            default_sheet=primary.default_sheet if primary else None,
            warnings=all_warnings,
            input_files=input_contexts,
            template_file=template_context.file_path if template_context else None,
            metadata={
                "data_file_count": len([item for item in input_contexts if item.role == "data"]),
                "template_file_name": template_context.file_path.name if template_context else None,
            },
        )
        return workbook, data_frames, template_frames
