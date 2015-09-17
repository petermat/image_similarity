# -*- coding: utf-8 -*-
"""

Installation of needed libraries
 
sudo apt-get install -y python-pip
sudo pip install PIL numpy
 
"""
from PIL import Image
import logging
import os, time, re, urllib
import urllib.request

format= '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)s() - %(message)s'
format= '%(asctime)s - %(filename)s:%(lineno)s - %(message)s'

log = logging.getLogger(__name__)
 
def image_similarity1(image_filepath1,image_filepath2):

    image1_filepath1 = image_filepath1
    image2_filepath2 = image_filepath2

    
    sim_list = begin_similarty_compare(image1_filepath1, image2_filepath2)
    return sim_list
 
def begin_similarty_compare(image_filepath1, image_filepath2):
    '''logger.debug("image file location: %s"%(download_base_directory ))
    url_pairs = re.split('#+',url_texts)
    urls = url_texts.strip().split()
    idx=0 # counter for downloaded image names
    for url_text in url_pairs:
        pair = url_text.strip().split()
        if not pair:
            continue
        filepath_url = []
        for url in pair:
            url = url.strip()
            filename = url.split('/')[-1]
            idx+=1
            filename = "%02.f-%s"%(idx,filename) # creates unique enumerated filenames
            #logger.debug("filename %s"%(filename))
            filepath = os.path.join(download_base_directory, filename)
            mkdir_p_filepath(filepath)
            if not os.path.exists(filepath):
                urllib.request.urlretrieve(url, filepath)
                logger.debug("downloading done")
            filepath_url.append( (filepath, url) )
 
        logger.debug("*"*20)
        logger.debug("compare images start")
        '''
        #image_filepath1, url1 = filepath_url[0][0], filepath_url[0][1]
        #logger.debug("image1: %s (%s)"%(get_filename(image_filepath1),url1))
        
        #image_filepath2, url2 = filepath_url[1][0], filepath_url[1][1]
        #logger.debug("image2: %s (%s)"%(get_filename(image_filepath2),url2))
 
    t1=time.time()
 
    similarity = image_similarity_bands_via_numpy(image_filepath1,image_filepath2)
 
    duration = "%0.1f"%((time.time() - t1)*1000)
    #log.info("DEBUG image_similarity: bands_via_numpy  => %s took %s ms"%(similarity,duration ))
    print("DEBUG image_similarity: bands_via_numpy  => %s took %s ms"%(similarity,duration ))
    similarity_bands_via_numpy = similarity
    t1=time.time()
 
    similarity = image_similarity_histogram_via_pil(image_filepath1,image_filepath2)
    similarity_histogram_via_pil = similarity
    
    duration = "%0.1f"%((time.time() - t1)*1000)
        #logger.debug("image_similarity_histogram_via_pil => %s took %s ms"%(similarity,duration ))
    print("DEBUG image_similarity: histogram_via_pil => %s took %s ms"%(similarity,duration ))
 
    t1=time.time()
 
    similarity = image_similarity_vectors_via_numpy(image_filepath1,image_filepath2)
    similarity_vectors_via_numpy = similarity
 
    duration = "%0.1f"%((time.time() - t1)*1000)
        #logger.debug("image_similarity_vectors_via_numpy => %s took %s ms"%(similarity,duration ))
    print("DEBUG image_similarity: vectors_via_numpy => %s took %s ms"%(similarity,duration ))
 
    t1=time.time()
 
    similarity = image_similarity_greyscale_hash_code(image_filepath1,image_filepath2)
    similarity_greyscale_hash_code = similarity
 
    duration = "%0.1f"%((time.time() - t1)*1000)
        #logger.debug("image_similarity_greyscale_hash_code => %s took %s ms"%(similarity,duration ))
    print("DEBUG image_similarity: greyscale_hash_code => %s took %s ms"%(similarity,duration ))
        
        
    print("DEBUG image_similarity: compare images finished")
    log.info("DEBUG image_similarity: compare images finished")
 
    return similarity_bands_via_numpy, similarity_histogram_via_pil, similarity_vectors_via_numpy, similarity_greyscale_hash_code 
 
def image_similarity_bands_via_numpy(filepath1, filepath2):
    import math
    import operator
    import numpy
    image1 = Image.open(filepath1)
    image2 = Image.open(filepath2)
 
    # create thumbnails - resize em
    image1 = get_thumbnail(image1)
    image2 = get_thumbnail(image2)
    
    # this eliminated unqual images - though not so smarts....
    if image1.size != image2.size or image1.getbands() != image2.getbands():
        return -1
    s = 0
    for band_index, band in enumerate(image1.getbands()):
        m1 = numpy.array([p[band_index] for p in image1.getdata()]).reshape(*image1.size)
        m2 = numpy.array([p[band_index] for p in image2.getdata()]).reshape(*image2.size)
        s += numpy.sum(numpy.abs(m1-m2))
    return s
 
def image_similarity_histogram_via_pil(filepath1, filepath2):
    from PIL import Image
    import math
    import operator
    from functools import reduce

    
    image1 = Image.open(filepath1)
    image2 = Image.open(filepath2)
 
    image1 = get_thumbnail(image1)
    image2 = get_thumbnail(image2)
    
    h1 = image1.histogram()
    h2 = image2.histogram()
 
    rms = math.sqrt(reduce(operator.add,  list(map(lambda a,b: (a-b)**2, h1, h2)))/len(h1) )
    return rms
 
def image_similarity_vectors_via_numpy(filepath1, filepath2):
    # source: http://www.syntacticbayleaves.com/2008/12/03/determining-image-similarity/
    # may throw: Value Error: matrices are not aligned . 
    from PIL import Image 
    from numpy import average, linalg, dot
    import sys
    
    image1 = Image.open(filepath1)
    image2 = Image.open(filepath2)
 
    image1 = get_thumbnail(image1, stretch_to_fit=True)
    image2 = get_thumbnail(image2, stretch_to_fit=True)
    
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # ValueError: matrices are not aligned !
    res = dot(a / a_norm, b / b_norm)
    return res
 
def image_similarity_greyscale_hash_code(filepath1, filepath2):
    # source: http://blog.safariflow.com/2013/11/26/image-hashing-with-python/
 
    image1 = Image.open(filepath1)
    image2 = Image.open(filepath2)
 
    image1 = get_thumbnail(image1, greyscale=True)
    image2 = get_thumbnail(image2, greyscale=True)
    
    code1 = image_pixel_hash_code(image1)
    code2 = image_pixel_hash_code(image2)
    # use hamming distance to compare hashes
    res = hamming_distance(code1,code2)
    return res
    
def image_pixel_hash_code(image):
    pixels = list(image.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join(map(lambda pixel: '1' if pixel < avg else '0', pixels))  # '00010100...'
    hexadecimal = int(bits, 2).__format__('016x').upper()
    return hexadecimal
 
def hamming_distance(s1, s2):
    len1, len2= len(s1),len(s2)
    if len1!=len2: 
        "hamming distance works only for string of the same length, so i'll chop the longest sequence"
        if len1>len2:
            s1=s1[:-(len1-len2)]
        else:
            s2=s2[:-(len2-len1)]
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])
 
def get_thumbnail(image, size=(128,128), stretch_to_fit=False, greyscale=False):
    " get a smaller version of the image - makes comparison much faster/easier"
    if not stretch_to_fit:
        image.thumbnail(size, Image.ANTIALIAS)
    else:
        image = image.resize(size); # for faster computation
    if greyscale:
        image = image.convert("L")  # Convert it to grayscale.
    return image
 
def mkdir_p_filepath(path):
    dirpath = os.path.dirname(os.path.abspath(path))
    mkdir_p(dirpath)
 
def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
 
def get_filename(path): 
    # cross plattform filename from a given path
    # source: http://stackoverflow.com/questions/8384737/python-extract-file-name-from-path-no-matter-what-the-os-path-format
    import ntpath
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
 
 
