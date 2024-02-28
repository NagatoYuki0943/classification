# AlexNet好处

> 图片中分为上下两部分,因为作者使用了两个GPU,学习的时候只用看下面即可

![image-20211103170617739](12XX AlexNet.assets/image-20211103170617739.png)

# 优点

> sigmoid, tanh激活函数: 计算复杂,梯度消失
>
> ReLU解决了这两个问题
>
> 

![image-20211103170640401](12XX AlexNet.assets/image-20211103170640401.png)

# 过拟合

![image-20211103170757153](12XX AlexNet.assets/image-20211103170757153.png)

![image-20211103171026477](12XX AlexNet.assets/image-20211103171026477.png)

# 卷积后的高宽计算

![image-20211103171106208](12XX AlexNet.assets/经过卷积后的矩阵高宽变化.png)

![image-20211103171137454](12XX AlexNet.assets/conv2d.png)

# 结构

![image-20211103172718983](12XX AlexNet.assets/image-20211103172718983.png)