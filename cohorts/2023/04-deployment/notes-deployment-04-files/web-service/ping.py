from flask import Flask


# give an identity to your web service
app = Flask('ping-pong')


# Define a route ('/ping') for the web service. The decorated function
# will be executed when that route is accessed.
@app.route('/ping',methods=['GET'])
def ping():
    return 'PONG'


if __name__=='__main__':
    # Run the Flask application and start the web service
    app.run(debug=True, host='0.0.0.0', port=9696)