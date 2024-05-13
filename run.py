from dotenv import load_dotenv
load_dotenv()

from scripts.flask_app import app

if __name__ == "__main__" :
    app.run(debug=False)