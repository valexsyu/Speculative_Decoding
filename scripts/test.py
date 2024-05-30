from collections import namedtuple


# SpModelOut = namedtuple(
#     "SpModelOut" , 
#     ["outputs", "accepted_count", "target_sample_count", "resample_count" ,"accepted_num" ]
# )
# SpDyGammaModelOut = namedtuple(
#     "SpDyGammaModelOut" , 
#     SpModelOut._fields + ("kl_div_out",)
# )


# def GG(x):
#     return SpDyGammaModelOut(x,x+1,x+2,x+3,x+4,x+5)

# breakpoint()
# print(GG(0)[1])


# from collections import namedtuple

# SpModelOut = namedtuple(
#     "SpModelOut", 
#     "output_tokens accepted_count target_sample_count resample_count accepted_num"
# )
# SpDyGammaModelOut = namedtuple(
#     "SpDyGammaModelOut", 
#     SpModelOut._fields + ("kl_div_out",)
# )

# def GG(x):
#     return SpDyGammaModelOut(
#         output_tokens = x,
#         accepted_count = x+1,
#         target_sample_count = x+1,
#         resample_count = x+1,
#         accepted_num = x+1,
#         kl_div_out = x+1
#     )

# # Example usage
# result = GG(5)
# breakpoint()
# print(result)

gamma_sequence = [10, 20, 30, 2, 3]
gamma = int(sum(gamma_sequence[-2:]) / len(gamma_sequence[-2:]) +0.5)
print(gamma)