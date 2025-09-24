# Copyright 2024 Your Name or Company Name
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law of agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimax plugin for LiveKit Agents

See [Plugin Docs URL - when available] for more information.
"""

# 1. 从我们的 tts.py (下一步将创建) 文件中导入核心 TTS 类
# 2. 导入版本号
from .tts import TTS
from .version import __version__

# 3. 声明哪些符号是这个包的公共 API
#    我们只实现 TTS, 所以只导出 TTS 和 __version__
__all__ = ["TTS", "__version__"]

from livekit.agents import Plugin

# 4. 导入我们刚刚创建的 logger
from .log import logger

# 5. 定义并注册插件
class MiniMaxPlugin(Plugin):
    def __init__(self) -> None:
        # super() 调用需要插件名称、版本、包路径和 logger
        super().__init__(__name__, __version__, __package__, logger)

# 将我们的插件实例注册到 LiveKit Agents 框架
Plugin.register_plugin(MiniMaxPlugin())

# 6. (可选但推荐) 清理文档，隐藏内部模块
#    这部分代码直接从 Cartesia 模板复制
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False