# server.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# 允许跨域，启用异步模式
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

@app.route('/')
def index():
    return render_template('index.html')

# 监听仿真脚本发来的 'sim_data' 事件
@socketio.on('sim_data')
def handle_sim_data(data):
    print(f"Received data: {data}")
    # 广播给所有连接的浏览器
    emit('update_plot', data, broadcast=True)

# 监听仿真开始/结束事件（可选，用于重置图表）
@socketio.on('sim_event')
def handle_sim_event(data):
    print(f"Event: {data}")
    emit('control_event', data, broadcast=True)

if __name__ == '__main__':
    print("Starting Web Server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000)