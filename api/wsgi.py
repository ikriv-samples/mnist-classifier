import sys
import traceback
import json

def notFound(path):
    return ("404 Not Found", f"Not found: '{path}'")

def nist(environ):
    APP_PATH='/var/www/dev.ikriv.com/api/mnist'

    method = environ['REQUEST_METHOD']
    if method != 'POST':
        return ("405 Method Not Allowed", "Method not allowed")
    input = json.loads(environ['wsgi.input'].read())
    image = input["image"]

    if APP_PATH not in sys.path:
        sys.path.insert(1, APP_PATH)
    import recognize

    prediction = recognize.inferDataUrl(image)
    return ("200 OK", str(prediction))

def handle(environ):
    path = environ['PATH_INFO']
    if path == "/mnist":
        return nist(environ)
    return notFound(path)

def exception():
    return ("500 Internal Server Error", traceback.format_exc())

def send(response, start_response):
    status, content, *tail = response
    contentType = tail[0] if tail else "text/plain"
    content = content.encode()
    response_headers = [('Content-type', contentType),
                        ('Content-Length', str(len(content)))]
    start_response(status, response_headers)
    return [content]

def application(environ, start_response):
    try:
        response = handle(environ)
    except:
        response = exception()
    return send(response, start_response)
