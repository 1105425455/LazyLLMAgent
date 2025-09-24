import json
from lazyllm import ReactAgent, fc_register, LOG, OnlineChatModule, WebModule

# 天气数据字典
WEATHER_DATA = {
    "北京": {
        "province": "北京市",
        "city": "北京",
        "publish_time": "2024-01-15 08:00:00",
        "weather": "晴天",
        "wind": "西北风 3-4级",
        "sunriseSunset": "日出: 07:30, 日落: 17:20",
        "temperature": "5°C / -3°C"
    },
    "上海": {
        "province": "上海市",
        "city": "上海",
        "publish_time": "2024-01-15 08:00:00",
        "weather": "阴天",
        "wind": "东南风 2-3级",
        "sunriseSunset": "日出: 07:00, 日落: 17:30",
        "temperature": "12°C / 8°C"
    },
    "广州": {
        "province": "广东省",
        "city": "广州",
        "publish_time": "2024-01-15 08:00:00",
        "weather": "雨天",
        "wind": "南风 4-5级",
        "sunriseSunset": "日出: 07:10, 日落: 18:00",
        "temperature": "18°C / 15°C"
    }
}

@fc_register("tool")
def get_weather(city: str):
    """
    天气查询。
    Args:
        city: 城市名（中文），当前仅支持北京、上海、广州
    Returns: 当地当天的天气信息
    """
    try:
        # 从字典中查找天气数据
        if city in WEATHER_DATA:
            res = WEATHER_DATA[city]
            return json.dumps(str(res), ensure_ascii=False)
        else:
            return f"抱歉，暂时无法查询到 {city} 的天气信息。目前支持的城市有：北京、上海、广州"
    except Exception as e:
        message = f"[Tool - get_weather] error occur, city: {city}, error: {str(e)[:512]}"
        LOG.error(message)
        return message

prompt = """
【角色】
你是一个出行建议助手，能够根据用户给定的城市名称主动查询天气信息，并给出出行建议。

【要求】
1. 根据用户的输入，调用工具查询当地天气情况
2. 城市名称为中文
3. 出行建议可以推荐一些活动
4. 目前支持的城市：北京（晴天）、上海（阴天）、广州（雨天）
"""

agent = ReactAgent(
    llm=OnlineChatModule(source='glm',model='glm-4.5',api_key='8df34f9ec1894e5f9fdf863c9f1aef6a.aSipXmwYch69ymaD',stream=False),
    tools=['get_weather'],
    prompt=prompt,
    stream=False
)

# 前端页面
w = WebModule(agent, port=8000, title="ReactAgent")
w.start().wait()
