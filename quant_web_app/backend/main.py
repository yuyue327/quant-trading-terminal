from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from data_loader import load_summary, load_stock_probs, load_ohlc, get_stock_list

app = FastAPI(title="Quant Expert System API", version="1.0")

# 允许跨域（React 前端访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/stocks")
async def api_get_stocks():
    """获取所有股票列表"""
    return {"stocks": get_stock_list()}

@app.get("/api/summary")
async def api_get_summary():
    """获取多股票绩效汇总"""
    data = load_summary()
    if isinstance(data, dict) and "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])
    return {"data": data}

@app.get("/api/ohlc/{stock}")
async def api_get_ohlc(stock: str):
    """获取 OHLC 数据"""
    data = load_ohlc(stock)
    if not data:
        raise HTTPException(status_code=404, detail=f"OHLC data not found for {stock}")
    return {"data": data}

@app.get("/api/probs/{stock}")
async def api_get_probs(stock: str):
    """获取预测概率 + 不确定性"""
    data = load_stock_probs(stock)
    if not data:
        raise HTTPException(status_code=404, detail=f"Probability data not found for {stock}")
    return {"data": data}

@app.get("/api/health")
async def health_check():
    return {"status": "online", "version": "1.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)