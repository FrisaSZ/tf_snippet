# 使用了Dataset模块的图的导出流程
使用tf1.4
没有用到Dataset模块时，在freeze的时候导出节点名称填写最终的运算会自动把最初的placeholder导出。
如果最初的placeholder是给Dataset用的，导出最终运算节点不会自动导出placeholder，需要显式的导出MakeIterator，后面加载模型时也需要对MakeIterator进行初始化。