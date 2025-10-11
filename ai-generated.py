import requests
import json

params = {
  'models': 'genai',
  'api_user': '1014340941',
  'api_secret': 'FDoe5NXWHyMDnXGg4zEvFo7yLSe5YChV'
}

def call(filename):
  files = {'media': open(filename, 'rb')}
  r = requests.post('https://api.sightengine.com/1.0/check.json', files=files, data=params)

  output = json.loads(r.text)
  return output["type"]["ai_generated"]