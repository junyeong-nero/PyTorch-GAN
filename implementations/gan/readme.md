# Generator

`nn.Linear`, `nn.batchnorm1d`를 이용해서 flatten한 features를 128, 256, 512, 1024까지 늘렸다가 다시 원래 이미지 사이즈까지 줄인다.

- $[B, \text{latent dim}] \rarr [B, C, W, H]$

```python

    ...
    self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img
    ...
```

# Discriminator

generator으로 부터 생성된 이미지의 fake/real을 구분한다. 

- $[B, C, W, H] \rarr [B, 1]$

```python
    self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
```


# Loss: BCELoss

$$
\operatorname*{min}_{G}V(D,G)=\mathbb{E}_{x\sim p_{a n}(x)}[\log D(x)]+\mathbb{E}_{z\sim p_{a}(z)}[\log(1-D(G(z)))].
$$

- BCELoss를 사용하는 이유: GAN은 generator로 부터 생성된 이미지를 discriminator가 adversarial 하게 fake/real를 판별하는 구조이다. fake/real를 판별하는 것이 0 또는 1의 binary output을 return 하기 때문에 BCELoss를 사용한다.

# Paper
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
