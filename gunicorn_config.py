# Gunicorn configuration file
import multiprocessing

# Server socket
bind = "0.0.0.0:${PORT:-8000}"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Timeout (in seconds)
timeout = 120

# Keep-alive
keepalive = 5
