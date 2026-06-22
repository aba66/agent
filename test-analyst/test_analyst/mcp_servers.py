from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .configuration import ConfigManager
from .file_loader import DataParserAgent
from .reporting import ReportGeneratorAgent


@dataclass
class MCPTool:
    name: str
    description: str
    handler: Callable[..., Any]


class LocalMCPServer:
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self.tools: dict[str, MCPTool] = {}
        self.resources: dict[str, Callable[[], Any]] = {}

    def register_tool(self, name: str, description: str, handler: Callable[..., Any]) -> None:
        self.tools[name] = MCPTool(name=name, description=description, handler=handler)

    def register_resource(self, name: str, loader: Callable[[], Any]) -> None:
        self.resources[name] = loader

    def call_tool(self, name: str, **kwargs) -> Any:
        if name not in self.tools:
            raise KeyError(f"{self.name} 中不存在工具：{name}")
        return self.tools[name].handler(**kwargs)

    def read_resource(self, name: str) -> Any:
        if name not in self.resources:
            raise KeyError(f"{self.name} 中不存在资源：{name}")
        return self.resources[name]()


class LocalMCPRegistry:
    def __init__(self) -> None:
        self._servers: dict[str, LocalMCPServer] = {}

    def register(self, server: LocalMCPServer) -> None:
        self._servers[server.name] = server

    def call_tool(self, server_name: str, tool_name: str, **kwargs) -> Any:
        if server_name not in self._servers:
            raise KeyError(f"未注册的 MCP Server：{server_name}")
        return self._servers[server_name].call_tool(tool_name, **kwargs)

    def read_resource(self, server_name: str, resource_name: str) -> Any:
        if server_name not in self._servers:
            raise KeyError(f"未注册的 MCP Server：{server_name}")
        return self._servers[server_name].read_resource(resource_name)

    def list_servers(self) -> list[str]:
        return sorted(self._servers.keys())


def build_local_mcp_registry(
    *,
    data_parser: DataParserAgent,
    report_agent: ReportGeneratorAgent,
    config_manager: ConfigManager,
) -> LocalMCPRegistry:
    registry = LocalMCPRegistry()

    file_server = LocalMCPServer("file-intake-mcp", "文件读取、sheet 摘要与候选结构识别")
    file_server.register_tool(
        "inspect_inputs",
        "读取一个或多个数据文件并输出结构摘要。",
        data_parser.load_many,
    )
    registry.register(file_server)

    rules_server = LocalMCPServer("rules-config-mcp", "读取字段别名、业务规则和权限配置")
    rules_server.register_resource("field_aliases", config_manager.field_aliases)
    rules_server.register_resource("business_rules", config_manager.business_rules)
    rules_server.register_resource("permissions", config_manager.permissions)
    rules_server.register_tool("get_field_aliases", "读取字段别名", lambda: config_manager.field_aliases())
    rules_server.register_tool("get_business_rules", "读取业务规则", lambda: config_manager.business_rules())
    registry.register(rules_server)

    excel_server = LocalMCPServer("excel-template-mcp", "Excel 模板读取、模板化导出和样式写回")
    excel_server.register_tool(
        "generate_reports",
        "使用模板和分析结果生成 Excel、图表、Markdown、PDF、Docx 等产物。",
        report_agent.generate,
    )
    registry.register(excel_server)

    artifact_server = LocalMCPServer("report-artifact-mcp", "产物输出与报告生成")
    artifact_server.register_tool(
        "generate_reports",
        "基于分析结果生成完整产物。",
        report_agent.generate,
    )
    registry.register(artifact_server)
    return registry
