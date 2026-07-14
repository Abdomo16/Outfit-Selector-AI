import requests

url = 'http://localhost:8000/wardrobe/upload'
files = {'file': ('sample_test.jpg', open('sample_test.jpg', 'rb'), 'image/jpeg')}

print("Sending request to server...")
response = requests.post(url, files=files)

print("Response Status Code:", response.status_code)
print("Response JSON:")
print(response.json())
