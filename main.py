from fastapi import FastAPI
import uvicorn
from mysite.api import check_number, check_clothes, cifar10


MY_models = FastAPI()

MY_models.include_router(check_number.check_image_router)
MY_models.include_router(check_clothes.clothing_check_router)
MY_models.include_router(cifar10.check_cifar_router)


if __name__ == '__main__':
    uvicorn.run(MY_models, host='127.0.0.1', port=8000)