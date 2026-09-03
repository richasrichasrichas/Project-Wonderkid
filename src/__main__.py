from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

class MeuApp(App):
    def build(self):
        # O Kivy carrega o arquivo meuapp.kv automaticamente!
        # Essa função só precisa retornar a base da interface
        return None # Como estamos usando o .kv, ele resolve sozinho

    def botao_clicado(self):
        # Acessando o texto que está na tela pelo ID
        label = self.root.ids.meu_texto
        label.text = "Você clicou! O back-end funcionou!"
        print("Lógica em Python processada com sucesso.")

if __name__ == '__main__':
    MeuApp().run()