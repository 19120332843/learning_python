import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.config import Config
 
Config.write()
 
class MainApp(App):
    def build(self):
        Window.fullscreen = 1
        layout = GridLayout(cols=8,spacing=3)
        layout.add_widget(Button(text="1"))
        layout.add_widget(Label(text="2"))
        layout.add_widget(Button(text="3"))
        layout.add_widget(Label(text="4"))
        layout.add_widget(Label(text="5")) # 如果不指定后缀则默认会自动查找ttf格式的字体
        layout.add_widget(Button(text="6"))
        layout.add_widget(Label(text="7")) # 别名指定
        layout.add_widget(Button(text="8")) # 别名指定
        return layout
 

if __name__ == '__main__':
  MainApp().run()