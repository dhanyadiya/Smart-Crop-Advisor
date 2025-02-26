from flask import Flask, render_template, request
import pandas as pd
import pickle
import cv2
import os
from werkzeug.utils import secure_filename
from utils.fertilizer import fertilizer_dic
from markupsafe import Markup


# Initialize Flask app
app = Flask(__name__)

# Load the trained models
with open('soil_classifier.pkl', 'rb') as f:
    soil_classifier = pickle.load(f)

with open('recommendation_model.pkl', 'rb') as f:
    recommendation_model = pickle.load(f)
# Load the pre-trained model pipeline
with open("fertilizer_model.pkl", "rb") as file:
    model= pickle.load(file)

# Load datasets
soil_data = pd.read_csv('soil.csv')
environment_data = pd.read_csv('environment.csv')
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pest recommendations mapping
pest_recommendations = {
    'rice': {
        'pests': ['Brown planthopper','White-backed planthopper'],
        'control_methods': ['Use neem oil','Introduce natural predators like dragonflies'
        ]
    },
    'maize': {
        'pests': ['Corn borer', 'Fall armyworm'],
        'control_methods': ['Use Bacillus thuringiensis (Bt)', 'Plant companion crops like marigold']
    },
    'cotton': {
        'pests': ['Cotton bollworm', 'Aphids'],
        'control_methods': ['Apply insecticidal soap', 'Use neem cake']
    },
    'coconut': {
        'pests': ['Coconut moth', 'Red palm weevil'],
        'control_methods': ['Use pheromone traps', 'Introduce beneficial insects']
    },
    'mango': {
        'pests': ['Mango fruit fly', 'Leaf cutter ants'],
        'control_methods': ['Use fruit fly traps', 'Apply organic repellents']
    },
    'orange': {
        'pests': ['Aphids', 'Citrus leaf miner'],
        'control_methods': ['Introduce ladybugs', 'Use insecticidal soap']
    },
    'apple': {
        'pests': ['Codling moth', 'Apple maggot'],
        'control_methods': ['Use pheromone traps', 'Apply neem oil']
    },
    'grapes': {
        'pests': ['Grape phylloxera', 'Mealybugs'],
        'control_methods': ['Use insecticidal soap', 'Introduce beneficial insects']
    },
    'banana': {
        'pests': ['Banana weevil', 'Nematodes'],
        'control_methods': ['Use organic nematicides', 'Introduce natural predators']
    },
    'pomegranate': {
        'pests': ['Pomegranate borer', 'Mealybugs'],
        'control_methods': ['Use insecticidal soap', 'Apply neem oil']
    },
    'blackgram': {
        'pests': ['Aphids', 'Whiteflies'],
        'control_methods': ['Use yellow sticky traps', 'Introduce natural predators']
    },
    'mungbean': {
        'pests': ['Aphids', 'Leafhoppers'],
        'control_methods': ['Use insecticidal soap', 'Plant trap crops']
    },
    'mothbeans': {
        'pests': ['Aphids', 'Pod borers'],
        'control_methods': ['Use neem oil', 'Introduce parasitic wasps']
    },
    'pigeonpeas': {
        'pests': ['Pod borer', 'Leafcutter bees'],
        'control_methods': ['Use Bacillus thuringiensis (Bt)', 'Plant trap crops']
    },
    'kidneybeans': {
        'pests': ['Bean weevil', 'Aphids'],
        'control_methods': ['Use diatomaceous earth', 'Apply neem oil']
    },
    'chickpea': {
        'pests': ['Chickpea pod borer', 'Aphids'],
        'control_methods': ['Use insecticidal soap', 'Introduce natural predators']
    },
    'watermelon': {
        'pests': ['Aphids', 'Cucumber beetles'],
        'control_methods': ['Use row covers', 'Apply neem oil']
    },
    'papaya': {
        'pests': ['Papaya fruit fly', 'Spider mites'],
        'control_methods': ['Use insecticidal soap', 'Introduce beneficial insects']
    },
    'coffee': {
        'pests': ['Coffee borer beetle', 'Leaf rust'],
        'control_methods': ['Use pheromone traps', 'Apply organic fungicides']
    }
}

fertilizer_dic = {
        'NHigh': """The N value of soil is high and might give rise to weeds.
        <br/> Please consider the following suggestions:

        <br/><br/> 1. <i> Manure </i> – adding manure is one of the simplest ways to amend your soil with nitrogen. Be careful as there are various types of manures with varying degrees of nitrogen.

        <br/> 2. <i>Coffee grinds </i> – use your morning addiction to feed your gardening habit! Coffee grinds are considered a green compost material which is rich in nitrogen. Once the grounds break down, your soil will be fed with delicious, delicious nitrogen. An added benefit to including coffee grounds to your soil is while it will compost, it will also help provide increased drainage to your soil.

        <br/>3. <i>Plant nitrogen fixing plants</i> – planting vegetables that are in Fabaceae family like peas, beans and soybeans have the ability to increase nitrogen in your soil

      <br/> <hr>""",

        'Nlow': """The N value of your soil is low.
        <br/> Please consider the following suggestions:
        <br/><br/> 1. <i>Add sawdust or fine woodchips to your soil</i> – the carbon in the sawdust/woodchips love nitrogen and will help absorb and soak up and excess nitrogen.

        <br/>2. <i>Plant heavy nitrogen feeding plants</i> – tomatoes, corn, broccoli, cabbage and spinach are examples of plants that thrive off nitrogen and will suck the nitrogen dry.

        <br/>3. <i>Water</i> – soaking your soil with water will help leach the nitrogen deeper into your soil, effectively leaving less for your plants to use.

        <br/>4. <i>Sugar</i> – In limited studies, it was shown that adding sugar to your soil can help potentially reduce the amount of nitrogen is your soil. Sugar is partially composed of carbon, an element which attracts and soaks up the nitrogen in the soil. This is similar concept to adding sawdust/woodchips which are high in carbon content.

       <br/> <hr>""",

        'PHigh': """The P value of your soil is high.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Avoid adding manure</i> – manure contains many key nutrients for your soil but typically including high levels of phosphorous. Limiting the addition of manure will help reduce phosphorus being added.

        <br/>2. <i>Use only phosphorus-free fertilizer</i> – if you can limit the amount of phosphorous added to your soil, you can let the plants use the existing phosphorus while still providing other key nutrients such as Nitrogen and Potassium. Find a fertilizer with numbers such as 10-0-10, where the zero represents no phosphorous.

        <br/>3. <i>Water your soil</i> – soaking your soil liberally will aid in driving phosphorous out of the soil. This is recommended as a last ditch effort.

        <br/>4. Plant nitrogen fixing vegetables to increase nitrogen without increasing phosphorous (like beans and peas).

        <br/>5. Use crop rotations to decrease high phosphorous levels <br/> <hr>""",

        'Plow': """The P value of your soil is low.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Bone meal</i> – a fast acting source that is made from ground animal bones which is rich in phosphorous.

        <br/>2. <i>Rock phosphate</i> – a slower acting source where the soil needs to convert the rock phosphate into phosphorous that the plants can use.

        <br/>3. <i>Phosphorus Fertilizers</i> – applying a fertilizer with a high phosphorous content in the NPK ratio (example: 10-20-10, 20 being phosphorous percentage).

        <br/>4. <i>Organic compost</i> – adding quality organic compost to your soil will help increase phosphorous content.

        <br/> <hr>""",

        'KHigh': """The K value of your soil is high</b>.
        <br/> Please consider the following suggestions:

        <br/><br/>1. <i>Loosen the soil</i> deeply with a shovel, and water thoroughly to dissolve water-soluble potassium. Allow the soil to fully dry, and repeat digging and watering the soil two or three more times.

        <br/>2. <i>Sift through the soil</i>, and remove as many rocks as possible, using a soil sifter. Minerals occurring in rocks such as mica and feldspar slowly release potassium into the soil slowly through weathering.

        <br/>3. Stop applying potassium-rich commercial fertilizer. Apply only commercial fertilizer that has a '0' in the final number field. Commercial fertilizers use a three number system for measuring levels of nitrogen, phosphorous and potassium. The last number stands for potassium. Another option is to stop using commercial fertilizers all together and to begin using only organic matter to enrich the soil.

        <br/>4. Mix crushed eggshells, crushed seashells, wood ash or soft rock phosphate to the soil to add calcium. Mix in up to 10 percent of organic compost to help amend and balance the soil.

      <br/> <hr>
        """,

        'Klow': """The K value of your soil is low.
        <br/>Please consider the following suggestions:

        <br/><br/>1. Mix in muricate of potash or sulphate of potash
        <br/>2. Try kelp meal or seaweed
        <br/>3. Try Sul-Po-Mag
        <br/>4. Bury banana peels an inch below the soils surface
        <br/>5. Use Potash fertilizers since they contain high values potassium
        """
    }

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.flatten()
    return image

def get_soil_type(soil_image):
    processed_image = preprocess_image(soil_image)
    processed_image = processed_image.reshape(1, -1)
    predicted_soil = soil_classifier.predict(processed_image)[0]
    print(predicted_soil)
    return predicted_soil 

def get_rainfall(area_name):
    try:
        environment = environment_data[environment_data['City'] == area_name].iloc[0]
        rainfall_min = environment['Rainfall_min']
        rainfall_max = environment['Rainfal_maxl']
        return rainfall_min, rainfall_max  
    except IndexError:
        return None, None  

def recommend_crop(soil_type, rainfall_min, rainfall_max):
    crop_data = pd.read_csv('crop_recommendation.csv')
    suitable_crops = crop_data[
        (crop_data['Soil'] == soil_type) & 
        (crop_data['rainfall'] <= rainfall_max) & 
        (crop_data['rainfall'] >= rainfall_min)
    ]
    
    if not suitable_crops.empty:
        crop_counts = suitable_crops['label'].value_counts(normalize=True) * 100
        crop_percentages = {crop: round(percentage, 2) for crop, percentage in crop_counts.items()}
        return crop_percentages
    else:
        return {"No suitable crop found": 100}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload.html')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        soil_image = request.files['image_path']
        district = request.form['districtSelect']
        city = request.form['city']
        
        filename = secure_filename(soil_image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        soil_image.save(image_path)
        
        soil_type = get_soil_type(image_path)
        rainfall_min, rainfall_max = get_rainfall(city)
        
        if rainfall_min is None or rainfall_max is None:
            return "Error: Could not find rainfall data for the selected area."
        
        crop_percentages = recommend_crop(soil_type, rainfall_min, rainfall_max)
        
        os.remove(image_path)
        
        return render_template('result.html', crop_percentages=crop_percentages)


@app.route('/crop_pest_info/<crop>')
def crop_pest_info(crop):
    pest_info = pest_recommendations.get(crop, {})
    return render_template('crop_pest_info.html', crop=crop, pest_info=pest_info)

# Routes for additional templates and functionality
@app.route('/pest.html')
def pest():
    return render_template('pest.html')

@app.route('/result')
def result():
    crops = request.args.get('crops')
    return render_template('result.html', crops=crops)

@app.route('/Contact.html')
def contact():
    return render_template('Contact.html')

@app.route('/rice')
def rice():
    return render_template('rice.html')

@app.route('/maize')
def maize():
    return render_template('maize.html')

@app.route('/cotton')
def cotton():
    return render_template('Cotten.html')

@app.route('/Coconut')
def Coconut():
    return render_template('Coconut.html')

@app.route('/mango')
def mango():
    return render_template('mango.html')

# render fertilizer recommendation form page


@ app.route('/black.html')
def fertilizer_recommendation():

    return render_template('black.html')


# fertilizer

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    # Load the trained model
   

    # Predict using the model
    prediction = model.predict([[N, P, K]])[0]

    # Access the dataset for the chosen crop
    df = pd.read_csv('Data/fertilizer.csv')
    crop_data = df[df['Crop'] == crop_name].iloc[0]
    nr, pr, kr = crop_data['N'], crop_data['P'], crop_data['K']

    # Calculate differences
    n_diff = nr - N
    p_diff = pr - P
    k_diff = kr - K

    # Determine deficient nutrients
    suggestions = []
    if abs(n_diff) > 20:
        suggestions.append(fertilizer_dic['NHigh'] if n_diff < 0 else fertilizer_dic['Nlow'])
    if abs(p_diff) > 10:
        suggestions.append(fertilizer_dic['PHigh'] if p_diff < 0 else fertilizer_dic['Plow'])
    if abs(k_diff) > 10:
        suggestions.append(fertilizer_dic['KHigh'] if k_diff < 0 else fertilizer_dic['Klow'])

    # Render result template with all recommendations
    if suggestions:
        response = Markup('<br/>'.join(suggestions))
    else:
        response = Markup("All nutrient levels are optimal.")

    return render_template('fertilizer-result.html', recommendation=response)

if __name__ == "__main__":
    app.run(debug=True)


