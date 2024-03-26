from helpers.losses import * #just convenient way


class WordModel(nn.Module):
    def __init__(self, pseudoCLIP_model, tokenizer, index_to_change=2, word_basis_start="a photo of a",
                 word_basis_end="", n_style_words=80, style_word_dim=512, device="cuda", style_words_to_load=None):
        '''
            model to train pseudowords with given phrase

        :param pseudoCLIP_model: clip model to use for encoding
        :param index_to_change: index of word that shall be a pseudoword >0 = in start_phrase, <0 = negative from end phrase [-2] for "e a b c d"  ="c"
        :param word_basis_start: prefix (poss including pseudoword dummy) for [class]
        :param word_basis_end: postfix for [class] (can contain pseudo dummy)
        :param n_style_words: number of stylewords to create for this template
        :param style_word_dim: = embedding size of encoding (512 ViT-B/16, 726 ViT-L/14)
        :param device: device to calculate, requires as style tokens are calculated in init (must be same as device of training)
        :param style_words_to_load: -- optional: init values for stylewords, if None init as gaussian(0,0.02)
        '''
        super(WordModel, self).__init__()
        if torch.is_tensor(style_words_to_load):
            self.style_words = style_words_to_load
        else:
            self.style_words = torch.nn.Parameter((torch.randn((n_style_words, style_word_dim))) * 0.02)
        self.style_words.requires_grad = True

        # for encoding
        self.pseudo_clip_encoder = pseudoCLIP_model
        for name, param in self.pseudo_clip_encoder.named_parameters():
            param.requires_grad = False

        self.index = index_to_change
        self.device = device
        self.phrase_start = word_basis_start
        self.phrase_end = word_basis_end
        self.tokenizer = tokenizer
        with torch.no_grad():
            self.pseudo_clip_encoder.eval()

            self.style_dummy_token = self.tokenizer(word_basis_start + " " + word_basis_end).to(
                self.device).detach()

    def forward(self, content_words):
        with torch.no_grad():
            style_features = self.pseudo_clip_encoder.encode_text(self.style_dummy_token,
                                                                  self.style_words,  # [:style_index+1],
                                                                  position_pseudo=self.index).to(torch.float32).to(
                self.device)

        for n_cont, content_word in enumerate(content_words):
            text = self.phrase_start + " " + content_word + " " + self.phrase_end
            sc_token = self.tokenizer(text).to(self.device)
            if n_cont == 0:
                style_content_features = self.pseudo_clip_encoder.encode_text(sc_token,
                                                                              self.style_words,  # [:style_index+1],
                                                                              position_pseudo=self.index).to(
                    torch.float32).to(self.device)
                style_content_features = style_content_features[:, None, :]
            else:
                sc_dummy = self.pseudo_clip_encoder.encode_text(sc_token, self.style_words,
                                                                position_pseudo=self.index).to(torch.float32).to(
                    "cuda")
                sc_dummy = sc_dummy[:, None, :]

                style_content_features = torch.cat(
                    (style_content_features, sc_dummy), 1)

        return [style_features, style_content_features]


class FullPseudoWordModel(nn.Module):
    def __init__(self, pseudoCLIP_model, tokenizer, class_word_index=1, pseudo_length=1, n_style_words_per_config=5, style_word_dim=512, device="cuda"):
        '''
            model to train pseudowords with a full pseudo basis (no real words except class names)
        :param pseudoCLIP_model: clip model to use for encoding
        :param tokenizer: which tokenizer to use (CLIP vs open_clip)
        :param class_word_index: position of [class] in string (in 0, pseudo_length), if == pseudo_length (e.g. =5), add after, at index 5 (6th word)
        :param pseudo_length: how many pseudowords to create for the prompt
        :param n_style_words_per_config: number of different pseudoword configurations (given amount = pseudo_length)
        :param style_word_dim: dim for style-words (768 ViT-L, 512, ViT-B)
        :param device:
        '''
        super(FullPseudoWordModel, self).__init__()

        self.style_words = torch.nn.Parameter((torch.randn((pseudo_length, n_style_words_per_config, style_word_dim), dtype=torch.float16)) * 0.02)
        self.style_words.requires_grad = True

        #for encoding
        self.pseudo_clip_encoder = pseudoCLIP_model
        for name, param in self.pseudo_clip_encoder.named_parameters():
            param.requires_grad = False

        self.cw_index = class_word_index
        self.pseudo_length = pseudo_length

        self.device = device
        self.n_style = n_style_words_per_config
        self.encode_dim = style_word_dim
        self.tokenizer = tokenizer
        with torch.no_grad():
            self.pseudo_clip_encoder.eval()

            self.style_dummy_token = self.tokenizer(" a"*self.pseudo_length).to(self.device).detach()
    def forward(self, content_words):
        for n_vector in range(self.n_style):
            style_feat = self.pseudo_clip_encoder.encode_text_multiple_pseudo(
                self.style_dummy_token,
                pseudowords=self.style_words[:, n_vector, :],
                positions_pseudo=[i + 1 for i in range(self.pseudo_length)])
            if n_vector == 0:
                style_features = style_feat
            else:
                style_features = torch.cat((style_features, style_feat))

        # -- STYLE CONTENT WORDS
        for n_cont, content_word in enumerate(content_words):
            text = "a "*self.cw_index + content_word + " a"*max(self.pseudo_length-self.cw_index, 0) #-- "a" is an example word, doesnt matter which,could also be "cat"
            sc_token = self.tokenizer(text).to(self.device)
            positions_pseudo = [i+1 for i in range(self.pseudo_length+1)] #+1 due to [start_token, a, b, c, d, end_token] in encode text
            positions_pseudo.remove(self.cw_index+1)

            for n_vector in range(self.n_style):
                sc_feat = self.pseudo_clip_encoder.encode_text_multiple_pseudo(
                                            sc_token,
                                            pseudowords=self.style_words[:, n_vector, :],
                                            positions_pseudo=positions_pseudo)

                if n_vector == 0:
                    sub_style_content_features = sc_feat
                else:
                    sub_style_content_features = torch.cat((sub_style_content_features, sc_feat))


            if n_cont == 0:
                style_content_features = sub_style_content_features[:, None,  :]
            else:
                style_content_features = torch.cat( (style_content_features, sub_style_content_features[:, None, :]), dim=1)

        return [style_features, style_content_features]


class Linear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, style_content_words):
        x = self.fc(style_content_words)
        return x, self.fc.weight.detach().clone()

