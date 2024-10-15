
class EmbeddingDataset(object):
    def __init__(self, inputs, labels, domain_labels):
        self.inputs = inputs
        self.labels = labels
        self.domain_labels = domain_labels

        assert len(self.inputs) == len(self.labels) == len(
            self.domain_labels), f"input, label, and domain label lengths don't match"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index], self.domain_labels[index]