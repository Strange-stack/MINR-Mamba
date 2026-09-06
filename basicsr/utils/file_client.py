class FileClient:
    """Read binary files from the local filesystem."""

    def __init__(self, backend='disk', **kwargs):
        if backend != 'disk':
            raise ValueError('Only the disk backend is supported.')
        if kwargs:
            raise ValueError(f'Disk backend does not accept extra options: {kwargs}')

    def get(self, filepath, client_key='default'):
        del client_key
        with open(str(filepath), 'rb') as file:
            return file.read()
