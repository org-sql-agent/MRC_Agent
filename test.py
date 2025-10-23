import requests
url='http://127.0.0.1:8000/generate'
payload={
  "prompt": "a small test",
  "width": 512,
  "height": 512,
  "steps": 20,
  "guidance_scale": 7.5,
  "seed": -1,
  "loras": []
}
r=requests.post(url,json=payload,timeout=300)
print('client status', r.status_code)
try:
    print('client json:', r.json())
except:
    print('client text len', len(r.text))