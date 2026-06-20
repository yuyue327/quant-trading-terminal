from fastapi.middleware.cors import CORSMiddleware

# 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://quant-frontend-0v0k.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
