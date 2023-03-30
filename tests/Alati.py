import unittest

from srbai.Alati.Transliterator import transliterate_cir2lat, transliterate_lat2cir


class AlatiTestovi(unittest.TestCase):
    def test_transliterate_cir2lat(self):
        lat = transliterate_cir2lat("Он је ишао на пр