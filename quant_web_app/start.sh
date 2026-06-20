#!/bin/bash
echo "🚀 启动量化专家系统后端 (FastAPI)..."
cd backend
pip3 install -r requirements.txt > /dev/null 2>&1
python3 main.py &
cd ..

echo "📦 启动前端 (React + Vite)..."
cd frontend
npm install > /dev/null 2>&1
npm run dev &
cd ..

echo "✅ 系统已启动！"
echo "🌐 前端访问: http://localhost:5173"
echo "🔗 后端 API: http://localhost:8000/docs"
wait