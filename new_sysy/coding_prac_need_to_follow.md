You've got it! When I write or refactor code for you, I strictly adhere to industry-standard software engineering principles to ensure the codebase remains maintainable, scalable, and easy to read. 

Here are the core principles I follow and what they mean:

### 1. DRY (Don't Repeat Yourself)
*   **What it means:** Every piece of knowledge or logic must have a single, unambiguous representation within a system. 
*   **How I use it:** If I see the same block of code written in three different places, I will extract it into a single reusable function or class. This prevents "copy-paste bugs" where you fix an issue in one place but forget to fix it in the others.

### 2. KISS (Keep It Simple, Stupid)
*   **What it means:** Systems work best when they are kept simple rather than made overly complex. 
*   **How I use it:** I avoid "clever" one-liners that are hard to read. I prefer writing clear, straightforward code that another developer can understand in 5 seconds. Complexity should only be added when absolutely necessary.

### 3. YAGNI (You Aren't Gonna Need It)
*   **What it means:** Always implement things when you actually need them, never when you just foresee that you *might* need them.
*   **How I use it:** I don't over-engineer solutions. For example, I won't build a massive database-driven configuration system if a simple JSON file perfectly solves the problem today. 

### 4. SOLID Principles (The Foundation of OOP)
If we are writing Object-Oriented code, I follow these 5 rules:
*   **S - Single Responsibility:** A class or function should do exactly *one* thing. (e.g., `QueryEngine` shouldn't also be managing database connections).
*   **O - Open/Closed:** Code should be open for extension (adding new features) but closed for modification (you shouldn't have to rewrite existing core logic to add a new feature).
*   **L - Liskov Substitution:** If you swap a parent class with its child class, the program shouldn't crash.
*   **I - Interface Segregation:** Don't force code to depend on methods it doesn't use. Keep interfaces small.
*   **D - Dependency Inversion:** High-level logic shouldn't depend on low-level details. Both should depend on abstractions (like using dependency injection).

### 5. Clean Code & Readability
*   **Self-Documenting Code:** I use highly descriptive variable and function names. `def fetch_user_data()` is vastly superior to `def get_data()`.
*   **Fail Fast:** If something is wrong (like a missing file or bad input), the code should raise an error immediately rather than failing silently 10 steps later.
*   **Explicit over Implicit:** I don't use "magic" variables. Everything the function needs should be passed as an argument.

Whenever we tackle a new script or refactor an old one (like fixing the modularity issues the agent pointed out in `query_engine.py`), I will automatically apply these rules to make sure your system is rock solid. Just let me know what you want to build or fix next!