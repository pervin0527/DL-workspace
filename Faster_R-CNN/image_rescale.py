import cv2

def rescale_image(image, size=600, max_size=1000):
    """
    이미지의 height, width 중 짧은 부분을 s, 긴 부분을 l 이라고 할 때,  
    s를 600의 값으로 resize한다. 그리고 l에는 B = (600 / s) 값을 곱해줘서 원본 이미지의 aspect-ratio를 유지하도록 한다.  
    - B값이 1000보다 큰 경우 600 X (1000 / B)를 곱해 600보다 작게 만든다.
    - 1000보다 작거나 같은 경우 s=600이고 l은 B로 rescaling 한다.
    """
    width, height = image.shape[:2]
    print("original image width and height", width, height)
    aspect_ratio = width / height
    print("original image aspect ratio", aspect_ratio)

    if width > height:
        min_part = height
    else:
        min_part = width

    if size / min_part > max_size:
        max_size = size * (max_size / (size / min_part))
    
    resized = cv2.resize(image, (max_size, size))
    return resized


min_size, max_size = 600, 1000
img_path = "./000001.jpg"
original_image = cv2.imread(img_path)
result_image = rescale_image(original_image, min_size, max_size)