import mesh_io
import torch

if __name__ == '__main__':
    eyelid_model_path = './data/eyelid_model.json'
    device = torch.device('cuda:0')
    bs = mesh_io.loadJsonAsBlendShape(eyelid_model_path, device)
    identity_coeffs = torch.zeros(
        (bs.identities.shape[0]), device=device)
    identity_coeffs[0] = 1.0
    expressions_coeffs = torch.zeros(
        (bs.expressions.shape[0]), device=device)
    expressions_coeffs[0] = 1.0
    # print(identity_coeffs, identity_coeffs.shape, bs.identities.shape)
    bs(identity_coeffs, expressions_coeffs)

    mesh_io.saveObj("tmp.obj", bs.morphed.to('cpu').detach().numpy().copy(
    ), [], [], bs.indices.to('cpu').detach().numpy().copy(), [], [], [])
