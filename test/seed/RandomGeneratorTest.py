from seed.generator.ImageRandomGenerator import ImageRandomGenerator
from seed.generator.TextRandomGenerator import TextRandomGenerator

# test ImageRandomGenerator
image_generator = ImageRandomGenerator((10, 10, 3))
print(image_generator.generate())

# test TextRandomGenerator
text_generator = TextRandomGenerator(50)
text = text_generator.generate()
print(type(text))
print(text)
