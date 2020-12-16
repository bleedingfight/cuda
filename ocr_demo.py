from aip import AipOcr
import os
""" 你的 APPID AK SK """
APP_ID = '21212393'
API_KEY = '3PQBmIGaKUX5V9CCUdpGG0X2'
SECRET_KEY = 'W1C7fZWnWiYUBhDPn0dFUFEY6SGzf4DO'
client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def save_info(info,savefile='CMakeLists.txt'):
    save_path = "/Users/biomind/Downloads/cuda"
    files = os.path.join(save_path,savefile)
    with open(files,'w') as f:
        for e in info:
            f.write(e['words']+"\n")
def predict(image):
    """ 调用通用文字识别（高精度版） """
    client.basicAccurate(image)
    """ 如果有可选参数 """
    options = {}
    options["detect_direction"] = "true"
    options["probability"] = "true"
    """ 带参数调用通用文字识别（高精度版） """
    r= client.basicAccurate(image, options)
    return r

def main():
    filepath = os.path.join('/Users/biomind/Downloads/cuda',"geluPlugin.png")
    image = get_file_content(filepath)
    r = predict(image)
    info = r['words_result']
    save_info(info,'gelu.txt')
if __name__ == "__main__":
    main()
