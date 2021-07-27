# CAMo
2019 보안프로젝트 - CAMO



## source_code (주요 파일 설명)
- static
    - assets, images : css, js, image 소스파일
    - facerec1, train : 사진 크롭 및 훈련 데이터 저장 폴더

- templates : html(index.html : 메인페이지 / viewer.html : 시청자페이지 / index_plus.html : 개인정보수집 안내 내용 추가 ver, 오류존재 / upload.html : add를 실행하지 않고 start를 할 때, 사진을 업로드하는 화면, 오류존재)

- setting : : flask - apache 연동 시 필요한 파일

- logs : flask - apache 연동 시 필요 폴더

- imoji : 이모티콘 ver 비식별화 / imoji.py 실행

- etc : 수정 파일

- main.py : camo 실행 파일

### main.py 실행
vgg_face_weights.h5 파일 필요
https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view

### imoji.py 실행
haarcascade_frontalface_default.xml 파일 필요
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
