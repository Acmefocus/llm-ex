# -*-*- exceptions.py -*-*-
class ToolError(Exception):
    """工具调用基础异常"""
    pass


class ToolNotFound(ToolError):
    """工具未找到异常"""
    pass


class ExecutionFailed(ToolError):
    """工具未找到异常"""
    pass



class ToolNotFoundError(ToolError):
    """工具未找到异常"""
    pass

class InvalidParamsError(ToolError):
    """参数验证失败异常"""
    pass

class ToolExecutionError(ToolError):
    """工具执行异常"""
    pass