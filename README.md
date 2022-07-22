# kanji-reader-python
* **WARNING**: this is an *early work*. I don't care about backward
compatibility and such things. Use it or not at your sole discretion.

A calligraphy reader for Japanese kanji, written in Python. It uses a
relatively dumb algorithm to detect kanji keypoints on an image, and match
them to a database.

## Usage
### Optical Character Recognition (kanji reading)

Simply call the script and an input image (PNG, JPEG, etc...) containing a
single kanji.
```bash
python main.py ./training/hand_written/two/人.png
```

The script will output various information, and return the best match in the
database.

### New Character Registration

 To add a new kanji to the database, you must provide the input image, say
 that we arre in learning mode (``--learning``) and give the number of strokes
 for this kanji (``--strokes``). A number of strokes too high is better than
 too few.

 For instance
```bash
python main.py --learning --strokes 3 --name 人 training/hand_written/two/人.png
```

You can also provide a ``--name``, it we simply make the output easier to
integrate in the database.


## Install
1. Clone this repository.
1. Install Python.
1. Install the required deoendencies: numpy and OpenCV for Python.

## Recognition algorithm
The algorithm used was made on purpose and do not rely on more serious research.
Let's say you want to identify the following image (kanji nekoà, cat).
![Kanji for cat (neko)](https://github.com/HugoFara/kanji-reader-python/raw/main/training/hand_written/eleven/neko.png)

The first step is to format the image: apply an Otsu threshold, crop the the black pixels and resize in 256*256px.
![Image formatting](https://github.com/HugoFara/kanji-reader-python/raw/main/doc/image_formatting.png)

Then, spheres are computed to find the biggest areas.

![Spheres growth](https://github.com/HugoFara/kanji-reader-python/raw/main/doc/spheres_evolution.gif)
![Spheres final state](https://github.com/HugoFara/kanji-reader-python/raw/main/doc/terminal_spheres.png)

In the end, we select the number of spheres that is at most two times the number of strokes for the kanji. It gives the following result.

![Decimated spheres](https://github.com/HugoFara/kanji-reader-python/raw/main/doc/decimated_spheres.png)


These spheres are matched against the database to find the closest kanji.

## Caveats
* This library will be split in two parts (learning and identification) if popular.
* Its sole purpose is to read characters in calligraphy.
