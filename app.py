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
        bp = request.args.get('select_bp') #  lấy dữ liệu từ trang home sau khi submit
        pred_dict = dict()
        pred_dict = {'0': 'DrugY', '1': 'drugA', '3':'drugC',  '2':'drugB', '4':'drugX' }
        #get model
        model = pickle.load(open('model.pkl','rb')) # load mô hình
        pred = model.predict([[age, sex, bp, cholesterol, na_to_k]]) # dự báo với dữ liệu lấy được
        print(pred)
        temp = ''
        for key,value in pred_dict.items():    
            if int(key) == pred:
                temp = value
        # Render the output in new HTML page
        return render_template('home.html', variety= temp) # biến temp là dữ liệu dự đoán được trả về trang output
    except:
        return 'Error'

if __name__=='__main__':
    app.run(debug=True)