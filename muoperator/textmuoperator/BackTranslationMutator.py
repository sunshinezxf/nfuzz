from muoperator.Mutator import Mutator
from googletrans import Translator


def back_translation(origin, backLang):
    translator = Translator(service_urls=[
        'translate.google.cn', ])
    srcL = translator.detect(origin)  # 语种识别
    trans = translator.translate(origin, src=srcL, dest=backLang)
    result = translator.translate(trans, src=backLang, destL=srcL)
    return result


def multiple_back_translation(origin):
    translator = Translator(service_urls=[
        'translate.google.cn', ])
    srcL = translator.detect(origin)
    trans1 = translator.translate(origin, src=srcL, dest='en')
    trans2 = translator.translate(trans1, src='en', dest='fr')
    result = translator.translate(trans2, src='fr', dest=srcL)
    return result


class BacKTranslationMutator(Mutator):
    """
        回译:
        seed为文本
        用googletrans进行翻译和回译
        当回译结果与原种子一样时，尝试复杂的回译
        复杂包括：
            更换另一种语种
            不同语种多次回译
    """

    def mutate(self, seed):
        mutant = back_translation(seed, 'en')
        if mutant == seed:
            mutant = back_translation(seed, 'fr')

        if mutant == seed:
            mutant = multiple_back_translation(seed)
        return mutant
