from abc import ABC, abstractmethod
from typing import Callable


class IngestionPlugin(ABC):
    """数据接入策略的抽象基类。

    垂直领域可继承此类实现自定义数据接入：
    - 电商：连接商品 API 定时拉取
    - 金融：接入实时行情 WebSocket
    - 法律：对接裁判文书数据库
    """

    @abstractmethod
    async def start(self, on_content: Callable):
        """启动数据接入。

        Args:
            on_content: async (text: str, source: str) -> None
                当有新增/更新内容时回调此函数。
        """
        ...


class EmptyIngestion(IngestionPlugin):
    """空接入策略，不做任何数据源监听。"""

    async def start(self, on_content: Callable):
        pass
