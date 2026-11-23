import torch
import torch.nn as nn
import torch.nn.functional as F
import math




def attention(Q, K, V, mask=None):
  # trin 1: Dot product, we want words x words. We have two matrix of words x numbers.
  # the order is important as rows = queries and columns = keys
  # We have to define that its the last 2 dimension to transpose such that we always transpose the words and numbers. This was an error when heads and teksts were introduced.
  dot_product_matrix = Q @ K.transpose(-2,-1)
  #print(dot_product_matrix.shape)

  # trin 2: scale by sqrt(dimension), we use Q.shape[-1] to make sure we get number of dimensions pr. word.
  scaled_dot_product = dot_product_matrix / math.sqrt(Q.shape[-1])

  # trin ekstra: mask, teksts have padding so they are the same length. this is important so the tensors have the same shape.
  # we do this, by making empty spaces = -infinty before the softmax, since softmax then makes them 0
  # This makes it so the words never pay attention to the padding. As softmax makes it 0%.
  # So the mask is a table with 1 if there is a word and 0 if this it padding.
  if mask is not None:
    scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, -math.inf)

  # trin 3: softmax af attention, we use dim=-1 to always get the last dimension, which is keys.
  softmax_attention = torch.softmax(scaled_dot_product, dim=-1)
  #print(softmax_attention)
  
  # trin 4: Every word becomes a weighted combinations of all the words values
  X_prime = softmax_attention @ V

  return X_prime, softmax_attention





class MultiHeadAttention(torch.nn.Module):
    def __init__(self, emb_dim, n_heads):
        # We have the variables from encoder.yaml

        # We here make sure not to mess with Pytorchs own init definition, by using super we run theirs first and then continue to ours. Like in the Prince notebook 11_2
        super(MultiHeadAttention, self).__init__()

        # emb_dim must be divisible by n_heads, because each head gets an equal share of the numbers per word. If not we wouyld get an error.
        assert emb_dim % n_heads == 0

        # By doing this we make sure the variable n_heads doesnt go losst after the init
        self.n_heads = n_heads
        # Same for the dimensions pr. head like we did in notebook 6.1
        self.head_dim = emb_dim // n_heads
    
        self.emb_dim = emb_dim

        # Define a Q, V, K and Output layer with the the input number emb_dim and same output number emb_dim.
        # As we saw in notebook 6.1, and the pytorch generates starting values. 
        self.linear1 = nn.Linear(emb_dim, emb_dim) # Q
        self.linear2 = nn.Linear(emb_dim, emb_dim) # K
        self.linear3 = nn.Linear(emb_dim, emb_dim) # V
        self.linear4 = nn.Linear(emb_dim, emb_dim) # Output

    def forward(self, x, mask=None):
        # Hvor x har formen (B, N, emb_dim), B = Batch (Antal tekster), N = Number of tokens (Længden af teksten)
        
    
        Q = self.linear1(x)
        K = self.linear2(x)
        V = self.linear3(x)


        # Hent størrelserne fra x: B = antal tekster, N = antal ord (bruges i view nedenfor)
        B, N, _ = x.shape

        # Split the dimension into multiple heads
        # Så i stedet for (B, N, emb_dim) så er det nu (B, N, n_heads, head_dim)
        Q = Q.view(B, N, self.n_heads, self.head_dim)

        # For det matcher attention definition, og bliver på formen (B, n_heads, N, head_dim)
        Q = Q.transpose(1,2)

        # Samme for K & V
        K = K.view(B, N, self.n_heads, self.head_dim)
        K = K.transpose(1,2)
        
        V = V.view(B, N, self.n_heads, self.head_dim)
        V = V.transpose(1,2)

        # Making sure the dimensions fit what our attention is expecting, by adding two ekstra dimensions. So we have (B, 1, 1, N)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)   

        # Defining outputs of our attention function
        # Out has shape: (B, n_heads, N, head_dim)
        out, attention_weights = attention(Q, K, V, mask)

        self.attention_weights = attention_weights

        # Nu har vi brugt det som opdelt i hoveder og vil gerne have det tilbage på formen som resten af modellen kan bruge.
        # Nu har out formen (B, n_heads, N, head_dim)
        # Så vi laver transpose og view tilbage igen.
        out = out.transpose(1,2) # Nu har out formen (B, N, n_heads, head_dim)
        out = out.contiguous() # fixes error before view som i notebook 6
        out = out.view(B, N, self.emb_dim) # sætter hovederne side om side igen, så formen er (B, N, emb_dim)

        # her sørger vi for tallene i out bliver ganget med vægtmatricen og får lagt bias til. Det sker inde i linear4
        out = self.linear4(out)
        
        return out



class FeedForward(torch.nn.Module):
    def __init__(self, emb_dim, feed_forward_dim):
        super(FeedForward, self).__init__()

        # Første linære lag. Udvider fra emb_dim til feed_forward_dim
        self.linear1 = nn.Linear(emb_dim, feed_forward_dim)

        # Andet linære lag. Fra feed_forward_dim tilbage til emb_dim
        self.linear2 = nn.Linear(feed_forward_dim, emb_dim)

    def forward(self, x):
        # vi har x på (B, N, emb_dim)

        # Linear lag 1
        x = self.linear1(x)
        # x bliver (B, N, feed_forward_dim)

        # Aktiverings lag, ReLU
        x = torch.relu(x)
        # x er stadig (B, N, feed_forward_dim)

        # Linear lag 2
        x = self.linear2(x)
        # x bliver (B, N, emb_dim)

        return x




class EncoderLayer(torch.nn.Module):
    def __init__(self, emb_dim, n_heads, feed_forward_dim):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(emb_dim, n_heads)
        self.feed_forward = FeedForward(emb_dim, feed_forward_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, mask=None):
        # x kommer med formen (B, N, emb_dim)

        # Equation 12.13 fra prince
        # trin 1: x + MhSa(x)
        x = x + self.attention(x, mask)

        # trin 2: layernorm(x)
        x = self.norm1(x)

        # trin 3: x + mlp(x)
        x = x + self.feed_forward(x)

        # trin 4: layernorm(x)
        x = self.norm2(x)

        return x




class Embeddings(torch.nn.Module):
    def __init__(self,vocab_size, emb_dim, max_len):
        super(Embeddings, self).__init__()

        # Vi opretter en matrix som fungere som ordbog med en række pr. token i hele ordforrådet.
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)

        # Vi opretter en lignende tabel men med en række for hver plads i teksten.
        self.position_embedding = nn.Embedding(max_len, emb_dim)

    def forward(self, token_ids):
        # Token_ids er (B, N)

        # B = hvor mange tekster, N = antal tokens
        B, N = token_ids.shape 

        # Tokens får positions nummer fra 0 indeksering og op
        positions = torch.arange(N, device = token_ids.device)

        # Slår først op i "ordbogen" hvad der står, så slår vi op hvor står det henne. Det lægger man sammen 
        # så hver vektor har en værdi som både er hvad og hvor det står.
        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        return x




class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, n_heads, emb_dim, feed_forward_dim, pooling_type="cls"):
        super(TransformerEncoder, self).__init__()

        # opretter varibel for pooling type, kan enten være CLS, mean eller max
        self.pooling_type = pooling_type

        # Opretter to tabeller.
        # En med en række for hvert ord i BERT ordbogen
        # En med en række for hver plads i teksten.
        # Det starter med tilfældige tal som læres under træningen.
        self.embeddings = Embeddings(vocab_size, emb_dim, max_len)

        # Opretter encoder layers på forhånd, som hver især har sine egne vægte.
        # I hvert lager sørger encoder layer for at ordene snakker sammen gennem attention og bagefter evalure hver for sig i feedforward laget.
        # nn.ModuleList så PyTorch kan se og træne lagene, ligesom i Week 6.
        self.layers = nn.ModuleList([EncoderLayer(emb_dim, n_heads, feed_forward_dim) for _ in range(n_layers)])

    def pool(self, x, attention_mask=None):



        if self.pooling_type == "cls":
            return x[:, 0]

        elif self.pooling_type == "mean":
                if attention_mask is None:
                    return x.mean(dim=1)

                # trin 1: masken er (B, N), men x er (B, N, emb_dim)
                # Vi tilføjer en ekstra dimenstion til enden så det bliver (B, N, 1), ligesom i MultiHeadAttention
                mask = attention_mask.unsqueeze(-1)

                # Vi er nødt til at lave den fra hel tal til decimaler for at kunne gange med x
                mask = mask.float()

                # trin 2: padding gøres til 0, så det ikke tæller med i summen
                x = x * mask

                # trin 3: lægger alle pladserne sammen for hver input. Så vi har (B, emb_dim)
                summed = x.sum(dim=1)

                #trin 4: tæl hvor mange ord der er i hver input (B, 1)
                n_words = mask.sum(dim=1)

                # trin 5: mean = sum / antal ord. Ikke N, fordi så var padding talt med og havde trukket resultatet ned.
                return summed / n_words
        

        elif self.pooling_type == "max":
                if attention_mask is None:
                    return x.max(dim=1).values

                # trin 1: masken er (B, N) og x er (B, N, emb_dim), så vi tilføjer en dimension og ender med (B, N, 1)
                mask = attention_mask.unsqueeze(-1)

                # trin 2: vi tager padding i x og sætter til -inf, så max aldrig kan vælge den. Præcis som i attention.
                x = x.masked_fill(mask == 0, -math.inf)

                # trin 3: for hver tal tager vi det ord med højest værdi.
                return x.max(dim=1).values



    def forward(self, token_ids, attention_mask=None):

        x = self.embeddings(token_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)

        return self.pool(x, attention_mask)
