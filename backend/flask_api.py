from flask import Flask, request, jsonify
from recommend_tracks import get_recommendations
from parse_rekordbox import fetch_rekordbox_data
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        current_track = data.get("current_track")
        num_recommendations = data.get("num_recommendations", 5)
        
        if not current_track:
            return jsonify({"error": "Missing 'current_track' parameter"}), 400
        
        recommendations = get_recommendations(current_track, num_recommendations)
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        logging.error(f"Error processing recommendation request: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/upload_rekordbox', methods=['POST'])
def upload_rekordbox():
    try:
        xml_file = request.json.get("xml_file")
        if not xml_file:
            return jsonify({"error": "Missing Rekordbox XML file parameter"}), 400
        
        track_data = fetch_rekordbox_data(xml_file)
        return jsonify({"tracks": track_data.to_dict(orient='records')})
    except Exception as e:
        logging.error(f"Error processing Rekordbox data: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
