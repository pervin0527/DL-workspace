import cv2

def resize_keep_aspect_ratio(image, size, max_size):
    w, h = image.shape[:2]
    print(f"original image width, height, aspect ratio : {w}, {h}, {w/h}")
    
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))

        ## 새로운 긴 부분 값이 1000보다 큰 경우 size변수 값을 재조정한다. 이로 인해 600보다 작을 수 있다.
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    ## 이미지의 짧은 부분이 이미 600(min size)와 같은 경우 원본 크기 그대로 반환.
    if (w <= h and w == size) or (h <= w and h == size):
        resized_image = cv2.resize(image, (h, w))
        return resized_image

    ## height, width 중 짧은 부분을 min_size로 변환하고
    ## 원본 이미지와 조정된 이미지간 aspect ratio를 유지하기 위한 계산.
    if w < h:
        ow = size
        oh = int(size * h / w) ## new_height = width_resize * original_height / original_width
    else:
        oh = size
        ow = int(size * w / h) ## new_width = height_resize * original_width / original_height

    resized_image = cv2.resize(image, (oh, ow))
    print(f"resized width, height, aspect ratio : {ow}, {oh}, {ow/oh}")
    return resized_image

# Example usage
input_image_path = 'input.jpg'  # Replace with your input image path
output_image_path = 'output.jpg'  # Replace with your output image path
max_size = 1000
min_size = 600

image = cv2.imread(input_image_path)
resized_image = resize_keep_aspect_ratio(image, min_size, max_size)
cv2.imwrite(output_image_path, resized_image)