from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 挂载前端打包后的静态文件夹
app.mount("/", StaticFiles(directory="quant_web_app/frontend/dist", html=True), name="static")

# 根路径返回首页
@app.get("/")
async def index_page():
    return FileResponse("quant_web_app/frontend/dist/index.html")
