from kivy.app import App
from kivy.graphics import Color,Rectangle
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

txtbuff = 'asd'
class RootWidget(FloatLayout):
    def __init__(self,**kwargs):
        super(RootWidget,self).__init__(**kwargs)
        self.add_widget(
            Button(
                text="Hello world",
                size_hint=(.5,.3),
                pos_hint={"center_x":.1,"center_y":.3}))
        
        self.add_widget(
            Button(
                text="Hello world",
                size_hint=(.3,.5),
                pos_hint={"center_x":.9,"center_y":.3}))

        self.add_widget(
            Label(
                text=txtbuff))

class MainApp(App):
    def build(self):
        self.root=root=RootWidget()
        root.bind(size=self._update_rect,pos=self._update_rect)
        with root.canvas.before:
            Color(0,0,1,1)   #设置背景颜色，（红，绿，蓝，透明度）
            self.rect=Rectangle(size=root.size,pos=root.pos)
        return root
    def _update_rect(self,instance,value):
        self.rect.pos=instance.pos
        self.rect.size=instance.size

if __name__=="__main__":
    MainApp().run()