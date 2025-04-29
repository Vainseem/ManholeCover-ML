#库文件
import uvicorn
from PIL import Image
import io
import torch
import logging
import requests
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from collections import Counter

#ml调用
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from model_training import Resnet_Network3,Resnet_Network2,Traditional_Network1,Regular_CNN,MaxPooling_CNN,ResidualBlock

#后端调用
#api_model存放api响应模型，db_model存放数据库模型，find存放数据库查询相关函数，
#tools存放数据库数据变动代码，emailsending基于stml独立完成用户意见反馈
from emailsending import send_mail_util
from api_models import UserInfoAPI, RecordAPI
from db_models import UserInfoDB, RecordDB
from find import find_area_code_by_name, find_areas_by_area_code,  \
     find_record_counts_by_address, find_userinfo_by_openid, find_record_by_id, \
     find_record_by_openid, find_record_by_address
from tools import add
from db import init_database

#微信小程序登录调用基本变量
WECHAT_APP_ID = 'wxff510f8e3bfa2d82'
WECHAT_APP_SECRET = '8a16e5c72a5b89943b7863581ee7e558'
WECHAT_SESSION = 'https://api.weixin.qq.com/sns/jscode2session'


app = FastAPI(docs_url="/docs", redoc_url=None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = {
    0: "broke",
    1: "circle",
    2: "good",
    3: "lose",
    4: "uncovered"
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    """startup event"""

    # 初始化数据库
    init_database()


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(60),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model = torch.load('Resnet_Network3_model1.0.pth')
model = model.to(device)
model.eval()

# Load the trained models
model1 = torch.load('Resnet_Network2_model2.0.pth')
model2 = torch.load('Resnet_Network2_model1.0.pth')
model3 = torch.load('Resnet_Network3_model1.0.pth')
model4 = torch.load('Traditional_Network1.0.pth')
model1 = model1.to(device)
model1.eval()
model2 = model2.to(device)
model2.eval()
model3 = model3.to(device)
model3.eval()
model4 = model4.to(device)
model4.eval()
logging.basicConfig(level=logging.INFO)

def predict_manhole_cover(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output1 = model1(image)
        output2 = model2(image)
        output3 = model3(image)
        output4 = model4(image)
        predictions = [output1, output2,output3, output4]
        outputs=Counter(predictions).most_common(1)[0][0]
        _, predicted = torch.max(outputs.data, 1)
        prediction_label = predicted.item()
    return class_names[prediction_label]

@app.post("/predict")
async def predict_manhole(image: UploadFile = File(...)):
    image_bytes = await image.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
        prediction = predict_manhole_cover(image)
        logging.info(f"Prediction: {prediction}")
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 用户反馈通道，发送qq邮件到邮箱
@app.get("/EmailSending")
async def Email(result: str = Depends(send_mail_util)):
    return result



# 通过code获取openid
@app.get('/openid')
async def openid_get(code: str):
    session = requests.get(WECHAT_SESSION,
                           params={'appid': WECHAT_APP_ID,
                                   'secret': WECHAT_APP_SECRET,
                                   'js_code': code,
                                   'grant_type': 'authorization_code'
                                   })
    return session.json()

# 通过openid检索用户信息
@app.get('/checkuserinfo')
async def checkuserinfo(openid: str):
    userinfo = find_userinfo_by_openid(openid)
    if userinfo is not None:
        return userinfo
    else:
        return 'nomessage'

# 通过id添加用户信息，如果发现openid重复则进行更新
@app.post('/adduserinfo')
async def adduserinfo(info: UserInfoAPI):
    addinfo = UserInfoDB(
        openid=info.openid,
        nickname=info.nickname,
        timestamp1=info.timestamp1,
        headimgurl=info.headimgurl,
        telephone=info.telephone
    )
    try:
        add(addinfo)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 通过openid查询record
@app.get('/record_openid')
async def area(openid=str):
    recordinfo = find_record_by_openid(openid)
    if recordinfo is not None:
        return recordinfo
    else:
        return 'nomessage'


# 通过address查询record
@app.get('/record_address')
async def area(address=str):
    recordinfo = find_record_by_address(address)
    if recordinfo is not None:
        return recordinfo
    else:
        return 'nomessage'

# 通过id查询record
@app.get('/record_id')
async def area(id: int):
    recordinfo = find_record_by_id(id)
    if recordinfo is not None:
        return recordinfo
    else:
        return 'nomessage'

# 上传record
@app.post('/record_post')
async def addrecordinfo(rec: RecordAPI):
    addrecordinfo = RecordDB(
        openid=rec.openid,
        timestamp2=rec.timestamp2,
        image=rec.image,
        status=rec.status,
        address=rec.address,
        remakes=rec.remakes
    )
    add(addrecordinfo)
    return 'OK'

# 通过name检索name_list和counts，输入区名，返回区下所有街道名及对应街道下记录条数
@app.get('/name_name_list')
async def location(name: str):
    narea_message = find_area_code_by_name(name)
    narea_code = narea_message.area_code

    if narea_code is None:
        return "nowhere"
    else:
        result = find_areas_by_area_code(narea_code)
        name_list = []
        merger_name_list = []
        counts = []
        for res in result:
            name_list.append(res.name)
            merger_name_list.append(res.merger_name)
        for address in merger_name_list:
            count = find_record_counts_by_address(address)
            counts.append(count)
        return name_list, counts


#########################################################################
if __name__ == "__main__":
    logging.info("Starting server process...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
