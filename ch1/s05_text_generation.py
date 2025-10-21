from transformers import pipeline

generator = pipeline(
    "text-generation",
    # Model size: 954M
    model = "Qwen/Qwen2.5-0.5B-Instruct"
)

request = "Write me a lemonade recipe"

prompt = f"User: {request}\nAssistant:"

response_list = generator(
    prompt,
    max_new_tokens = 200,
    num_return_sequences = 2
)

for i in range(len(response_list)):
    r = response_list[i]
    generated_text = r["generated_text"]
    response = generated_text.split("\nAssistant:")[1]
    print(f"Response {i + 1}:")
    print(response)
    print('====================')

# Response 1:
# Sure! Here's a simple recipe for lemonade:
#
# Ingredients:
# - 1 cup sugar
# - 1/2 cup water
# - 3 cups freshly squeezed orange juice (or any other favorite fruit juice)
# - 1 teaspoon lemon zest
#
# Instructions:
# 1. In a large saucepan, combine the sugar and water.
# 2. Bring to a boil over medium heat.
# 3. Reduce heat to low and simmer for about 5 minutes, or until the sugar has dissolved.
# 4. Remove from heat and let it cool slightly.
# 5. Pour into glasses and stir with ice cubes.
# 6. Add the orange juice and lemon zest.
#
# Enjoy your delicious homemade lemonade! #lemonaderecipe #sugar #juice #orangejuice #fruitjuice #lunch #breakfast
# #lunchtime #afternoontea #dinner #lunchtimedrink #juiceconvenience #healthydrink #healthyl
# ====================
# Response 2:
# Sure, here's a simple recipe for lemonade:
#
# Ingredients:
# - 1 cup sugar
# - 2 cups water
# - 1 tablespoon lemon juice
# - Ice cubes
#
# Instructions:
# 1. In a medium saucepan, combine the sugar and water.
# 2. Bring to a boil over medium heat.
# 3. Reduce the heat to low and let simmer for about 5 minutes, or until the mixture has thickened slightly.
# 4. Remove from heat and stir in the lemon juice.
# 5. Stir in ice cubes.
# 6. Pour into glasses and serve immediately.
#
# Enjoy your refreshing lemonade! 🍽️☕️ #lemonaderecipe #sugarandwater #icecreamrecipes #foodie #cookingtips #lunch
# #afternoontea #healthylifestyle #lemonade #sweetness #sourness #fruitjuice #fruitflavoreddrinks #healthylifestyle
# #nutritiontips #
# ====================
