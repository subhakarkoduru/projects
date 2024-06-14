import http.server
import ssl

# Specify the port you want to use
port = 8080

# Specify the directory where your files are located
directory = '.'  # Change this to the directory containing your files

# Specify the path to your SSL certificate and private key
certfile = 'localhost.pem'
keyfile = 'localhost-key.pem'

# Create an HTTP server
httpd = http.server.HTTPServer(('10.0.0.49', port), http.server.SimpleHTTPRequestHandler)
# If testing locally, you might want to use 'localhost' instead of '10.0.0.204'
# httpd = http.server.HTTPServer(('localhost', port), http.server.SimpleHTTPRequestHandler)

# Create an SSL context
context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain(certfile=certfile, keyfile=keyfile)  # Load your cert and key files

# Wrap the server's socket with SSL using the context
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print(f"Serving on https://10.0.0.204:{port}")
httpd.serve_forever()
