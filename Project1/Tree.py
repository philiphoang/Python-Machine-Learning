class Tree:

    def __init__(self, name='root', classifier = None):
        self.children = []
        self.name = name
        self.classifier = classifier
        self.data = {}

    def set_data(self, data):
        self.data = data

    def set_name(self, name):
        self.name = name

    def set_classifier(self, classifier):
        self.classifier = classifier

    def __repr__(self):
        return str(self.name)

    def add_child(self, tree):
        self.children.append(tree)

    def remove_child(self, child):
        self.children.remove(child)

    def clear_children(self, tree):
        self.children = []

    def isLeaf(self):
        return self.children.count(None) == len(self.children)

    def show(self, ind = 0):
        indent = ''
        for i in range(ind):
            indent = indent + ' | '

        if self.isLeaf():
            print(indent, self.name, '-->', self.classifier, self.data)
        else:
            print(indent, self.name, 'is a parent with children:',self.children , self.data)

            for child in self.children:
                child.show(ind + 1)
