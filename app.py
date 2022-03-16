from flask import Flask, request, render_template
import pickle
# Initialize the flask class and specify the templates directory
app = Flask(__name__)

# Default route set as 'home'
@app.route('/')
def home():
    return render_template('home.html') # Render home.html

@app.route('/classify',methods=['GET'])
def classify_type():
    try:
        age = request.args.get('age') # 
        na_to_k = request.args.get('na_to_K') # 
        sex = request.args.get('select_sex') # 
        cholesterol = request.args.get('select_cholesterol') # 
        bp = request.args.get('select_bp') # 
        pred_dict = dict()
        pred_dict = {'0': 'DrugY', '1': 'drugA', '2':'drugC',  '3':'drugB', '4':'drugX' }
        #get model
        model = pickle.load(open('model.pkl','rb'))
        pred = model.predict([[age, na_to_k, sex, cholesterol, bp]])
        temp = ''
        for key,value in pred_dict.items():    
            if int(key) == pred:
                temp = value
        # Render the output in new HTML page
        return render_template('output.html', variety= temp)
    except:
        return 'Error'

if __name__=='__main__':
    app.run(debug=True)