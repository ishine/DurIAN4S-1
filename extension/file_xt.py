import os 

def duplicate_path(path):
    duplicated_path = path
    n = 0
    while os.path.exists(duplicated_path):
        n += 1
        duplicated_path = path + '_' + str(n)
    
    os.makedirs(duplicated_path)

    return duplicated_path

class FileXT(object):
    def __init__(self, *args):
        path_list = []
        self.ext = ''
        for path in args:
            for p in path.split('/'):
                if p == '.':
                    p = os.getcwd()
                    path_list.append(p)
                elif p == '..':
                    p = os.path.dirname(os.getcwd())
                    path_list.append(p)
                elif len(p) > 0 and p[0] == '.':
                    self.ext = p
                else: 
                    if '.' in p:
                        self.ext = '.' + p.split('.')[-1]
                        p = p.split('.')[0]
                    
                    path_list.append(p)

        filestem = '/'.join(path_list)
        self.basename = os.path.basename(filestem) + self.ext
        self.basestem = os.path.basename(filestem) 
        self.filepath = os.path.dirname(filestem)
        self.filename = filestem + self.ext
        self.filestem = filestem

    def create_path(self, action="duplicate", verbose=True):
        created_path = self.filepath
        if os.path.exists(self.filepath):
            if action is "override":
                os.makedirs(self.filepath)
            elif action is "duplicate":
                created_path = duplicate_path(self.filepath)
            elif action is "error":
                raise AssertionError("\'%s\' already exists." % self.filepath)
            else: 
                raise AssertionError("Invalid action is used.")
        else:
            os.makedirs(self.filepath)

        if verbose:
            print("\'%s\' is created." % (created_path))

        return created_path