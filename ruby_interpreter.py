###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################

#token types
PLUS='PLUS'
MINUS='MINUS'
MUL='MUL'
DIV='DIV'
MOD='MOD'
LPAREN='('
RPAREN=')'
APOS='"'
INTEGER='INTEGER'
REAL='REAL'
EOF='EOF'
STR='STR'
TRUE='TRUE'
FALSE='FALSE'
EQUAL='=='
NOT='!='
GRET='>'
LEST='<'
GRE='>='
LESE='<='
ASSIGN='='
SEMI=';'
ID='ID'
COMMA=','
DOT='.'
FOR='for'
WHILE='while'
IF='if'
ELSE='else'
ELSIF='elsif'
END='end'

class Token(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        """String representation of the class instance.
        Examples:
            Token(INTEGER, 3)
            Token(PLUS, '+')
            Token(EQUAL, '==')
        """
        return 'Token({type}, {value})'.format(
            type=self.type,
            value=repr(self.value)
        )

    def __repr__(self):
        return self.__str__()


RESERVED_KEYWORDS={
    'if': Token(IF, 'if'),
    'elsif': Token(ELSIF, 'elsif'),
    'else': Token(ELSE, 'else'),
    'while': Token(WHILE, 'while'),
    'end': Token(END, 'end'),
    'true': Token(TRUE, 'true'),
    'false': Token(FALSE, 'false')
}


class Lexer(object):
    def __init__(self, text):
        # string input: "2+3*4"
        self.text = text
        # pos is a pointer to the current character of 'text'
        self.pos = 0
        self.line = 1
        # current token 
        self.current_token = None
        self.current_char = self.text[self.pos]

    def error(self):
        raise Exception('Invalid syntax')

    def advance(self):
        """Advance the pos pointer and set the current_char variable."""
        self.pos += 1

        if self.pos > len(self.text) - 1:      #This indicates EOF or end of line
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def peek(self, n):
        """Peeks the character n characters away from the current current character."""
        peek_pos = self.pos + n
        if peek_pos > len(self.text) - 1:      #Indicates EOF
            return None
        else:
            return self.text[peek_pos]

    def skip_white_space(self):
        """Skips white space in the input."""
        while self.current_char is not None and self.current_char.isspace():
            if self.current_char == '\n':
                self.line += 1
            self.advance()

    def skip_comment(self):
        """Skips comments in the input."""
        while self.current_char not in ['\n','\x00']:
            self.advance()

    def number(self):
        """Returns an integer(multidigit) or a float that is read from the input"""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        if self.current_char == '.':
            result += self.current_char
            self.advance()

            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()

            token = Token('REAL', float(result))

        else:
            token = Token('INTEGER', int(result))

        return token

    def string(self):
        result = ''
        self.advance()
        while self.current_char is not None and self.current_char != '"':
            result += self.current_char
            self.advance()
        self.advance()

        token = Token('STR', result)
        return token

    def _id(self):
        """Handle identifiers and reserved keywords"""
        result = ''
        while self.current_char is not None and self.current_char.isalnum() or self.current_char=='_':
            result += self.current_char
            self.advance()

        token = RESERVED_KEYWORDS.get(result, Token(ID, result))
        return token

    def get_next_token(self):
        """Lexical analyzer (also known as scanner or tokenizer)
        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_white_space()
                continue

            if self.current_char == '#':
                self.advance()
                self.skip_comment()
                continue

            if self.current_char.isalpha():
                return self._id()

            if self.current_char.isdigit():
                return self.number()

            if self.current_char == '=' and self.peek(1)!='=':
                self.advance()
                return Token(ASSIGN, '=')

            if self.current_char == '=' and self.peek(1)=='=':
                self.advance()
                self.advance()
                return Token(EQUAL, '==')

            if self.current_char == '!' and self.peek(1)=='=':
                self.advance()
                self.advance()
                return Token(NOT, '!=')

            if self.current_char == '>' and self.peek(1)!='=':
                self.advance()
                return Token(GRET, '>')

            if self.current_char == '<' and self.peek(1) == '=':
                self.advance()
                self.advance()
                return Token(LESE, '<=')

            if self.current_char == '>' and self.peek(1) == '=':
                self.advance()
                self.advance()
                return Token(GRE, '>=')

            if self.current_char == '<' and self.peek(1) != '=':
                self.advance()
                return Token(LEST, '<')

            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')

            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')

            if self.current_char == '*':
                self.advance()
                return Token(MUL, '*')

            if self.current_char == '/':
                self.advance()
                return Token(DIV, '/')

            if self.current_char == '%':
                self.advance()
                return Token(MOD, '%')

            if self.current_char == ';':
                self.advance()
                return Token(SEMI, ';')

            if self.current_char == ',':
                self.advance()
                return Token(COMMA, ',')

            if self.current_char == 'i' and self.peek(1) == 'f':
                self.advance()
                self.advance()
                return Token(IF, 'if')

            if self.current_char == 'e' and self.peek(1) == 'l' and self.peek(2) == 's' and self.peek(3) == 'e':
                for x in range (0,4):
                    self.advance()
                return Token(ELSE,'else')

            if self.current_char == 'e' and self.peek(1) == 'l' and self.peek(2) == 's' and self.peek(3) == 'i' and self.peek(4) == 'f':
                for x in range (0,5):
                    self.advance()
                return Token(ELSIF,'elsif')

            if self.current_char == 'w' and self.peek(1) == 'h' and self.peek(2) == 'i' and self.peek(3) == 'l' and self.peek(4) == 'e' and self.peek(5):
                for x in range (0,5):
                    self.advance()
                return Token(WHILE, 'while')

            if self.current_char == 'e' and self.peek(1) == 'n' and self.peek(2) == 'd':
                for x in range (0, 3):
                    self.advance()
                return Token(END, 'end')

            if self.current_char == 't' and self.peek(1) == 'r' and self.peek(2)=='u' and self.peek(3)=='e':
                for x in range (0,4):
                    self.advance()
                return Token(TRUE, 'true')

            if self.current_char == 'f' and self.peek(1) == 'a' and self.peek(2)=='l' and self.peek(3)=='s' and self.peek(4)=='e':
                for x in range (0,5):
                    self.advance()
                return Token(FALSE, 'false')

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')

            if self.current_char == '.':
                self.advance()
                return Token(DOT, '.')

            if self.current_char == '"':
                self.advance()
                return Token(APOS,'"')

            self.error()

        return Token(EOF, None)


###############################################################################
#                                                                             #
#  PARSER                                                                     #
#                                                                             #
###############################################################################
class AST(object):
    pass


class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class UnaryOp(AST):
    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr


class Compound(AST):
    def __init__(self):
        self.children = []


class Assign(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right

class Var(AST):
    """Var node is constructed from ID token"""
    def __init__(self,token):
        self.token=token
        self.value=token.value

class If(AST):
    def __init__(self, condition, body, rest):
        self.condition = condition
        self.body = body
        self.rest = rest

class Else(AST):
    def __init__(self, body):
        self.body=body


class While(AST):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class NoOp(AST):
    pass

class Parser(object):
    def __init__(self,lexer):
        self.lexer=lexer
        #set current token to the first token taken from the input
        self.current_token=self.lexer.get_next_token()

    def error(self):
        raise Exception('invalid syntax')
    
    def eat(self,token_type):
        """Compares the current token type with the passed token type"""
        #If these two match then eat the current token and set it to the next token
        #Else raise an exception
        if self.current_token.type==token_type:
            self.current_token=self.lexer.get_next_token()
        else:
            self.error()

    def program(self):
        """program : compound_statement"""
        node=self.compound_statement()
        return node

    def compound_statement(self):
        """compound_statement : statement_list"""
        nodes = self.statement_list()
        root=Compound()
        for node in nodes:
            root.children.append(node)
        return root

    def statement_list(self):
        """statement_list : statement | statement SEMI statement_list"""
        node = self.statement()
        results = [node]

        while self.current_token.type != EOF:
            results.append(self.statement())

        return results

    def statement(self):
        """
        statement : compound_statement | assignment_statement | if_statement | elsif_statement| else statement | while statement| empty
        """
        if self.current_token.type==ID:
            node=self.assignment_statement()
        elif self.current_token.type==IF:
            node=self.if_statement()
        elif self.current_token.type==ELSIF:
            node=self.elsif_statement()
        elif self.current_token.type==ELSE:
            node=self.else_statement()
        elif self.current_token.type==WHILE:
            node=self.while_statement()
        else:
            node=self.conditional_statement()
        return node

    def assignment_statement(self):
        """assignment_statement : variable ASSIGN expr"""
        left = self.variable()
        token = self.current_token
        self.eat(ASSIGN)
        right = self.expr()
        node = Assign(left, token, right)
        return node

    def if_statement(self):
        """
        if_statement : IF conditional_statement  statement_list  (elsif_statement | else_statement | empty) END
        """
        self.eat(IF)
        condition = self.conditional_statement()
        body=[]
        rest=[]
        while self.current_token.type!=ELSIF and self.current_token.type!=ELSE and self.current_token.type!=END:
            body.append(self.statement())
        if self.current_token.type==ELSIF:
            rest = self.elsif_statement()
        if self.current_token.type==ELSE:
            rest = self.else_statement()
        node = If(condition, body, rest)
        return node

    def while_statement(self):
        """
        while_statement : WHILE conditional_statement statement_list END
        """
        self.eat(WHILE)
        condition=self.conditional_statement()
        body=[]
        while self.current_token.type!=END:
            body.append(self.statement())
        self.eat(END)
        node=While(condition,body)
        return node
    
    def variable(self):
        """
        variable : ID
        """
        node = Var(self.current_token)
        self.eat(ID)
        return node

    def empty(self):
        """An empty production"""
        return NoOp()

    def elsif_statement(self):
        """
        elsif_statement : ELSIF conditional_statement statement_list (else | empty)
        """
        self.eat(ELSIF)
        elsif_condition=self.conditional_statement()
        elseif_body=[]
        rest=[]
        while self.current_token.type!=ELSE:
            elseif_body.append(self.statement())
        if self.current_token.type==ELSE:
            rest=self.else_statement()
        node=If(elsif_condition, elseif_body,rest)
        return node

    def else_statement(self):
        """else_statement : ELSE statement_list """
        self.eat(ELSE)
        else_body=[]
        while self.current_token.type!=END:
            else_body.append(self.statement())
        self.eat(END)
        node = Else(else_body)
        return node

    def conditional_statement(self):
        """conditional_statement : expr (EQUAL|GRE|NOT|GRET|LESE|LEST) expr"""
        node=self.expr()
        while self.current_token.type in (EQUAL,GRE,NOT,GRET,LESE,LEST):
            token=self.current_token()
            self.eat(token.type)
            node=BinOp(node,token,self.expr())
        return node

    def expr(self):
        """
        expr : term ((PLUS | MINUS) term)*
        """
        node = self.term()
        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
            elif token.type == MINUS:
                self.eat(MINUS)

            node = BinOp(node,token,self.term())
        return node

    def term(self):
        """term : factor ((MUL | DIV | MOD | EQUAL | NOT | LEST | GRET | LESE | GRE) factor)*"""
        node = self.factor()
        while self.current_token.type in (MUL,DIV, MOD,EQUAL,NOT,LEST,GRET,LESE,GRE):
            token = self.current_token
            if token.type == MUL:
                self.eat(MUL)
            elif token.type == DIV:
                self.eat(DIV)
            elif token.type == MOD:
                self.eat(MOD)
            elif token.type == EQUAL:
                self.eat(EQUAL)
            elif token.type == NOT:
                self.eat(NOT)
            elif token.type == LEST:
                self.eat(LEST)
            elif token.type == GRET:
                self.eat(GRET)
            elif token.type == LESE:
                self.eat(LESE)
            elif token.type == GRE:
                self.eat(GRE)
            node = BinOp(node,token,self.factor())

        return node

    def factor(self):
        """factor : PLUS factor
                  | MINUS factor
                  | INTEGER
                  | REAL
                  | LPAREN expr RPAREN
                  | variable
        """
        token = self.current_token
        if token.type == INTEGER:
            self.eat(INTEGER)
            return Num(token)
        elif token.type == REAL:
            self.eat(REAL)
            return Num(token)
        elif token.type == PLUS:
            self.eat(PLUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == MINUS:
            self.eat(MINUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            self.eat(RPAREN)
            return node
        elif token.type == ID:
            self.eat(ID)
            return Var(token)
        elif token.type == STR:
            self.eat(STR)
            return Num(token)
        else:
            node = self.variable()
            return node

    def parse(self):
        """
        program : compound_statement
        compound_statement : statement_list
        statement_list : statement | statement SEMI statement_list
        statement : compound_statement | assignment_statement |  if_statement | elsif_statement| else statement | while statement| empty
        assignment_statement : variable ASSIGN expr
        if_statement : IF conditional_statement  statement_list  (elsif_statement | else_statement | empty) END
        elsif_statement : ELSIF conditional_statement statement_list (else | empty)
        else_statement : ELSE statement_list
        while_statement : WHILE conditional_statement statement_list END
        conditional_statement : expr (EQUAL|GRE|NOT|GRET|LESE|LEST) expr
        expr : term ((PLUS | MINUS) term)*
        term : factor ((MUL | DIV | MOD | EQUAL | NOT | LEST | GRET | LESE | GRE) factor)*
        factor : PLUS factor
                  | MINUS factor
                  | INTEGER
                  | REAL
                  | LPAREN expr RPAREN
                  | variable
        variable : ID
        """
        node = self.program()
        while self.current_token.type != EOF:
            self.error()
        return node


class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))

###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################


class Interpreter(NodeVisitor):
    
    GLOBAL_MEMORY = {}
    
    def __init__(self, tree):
        self.tree = tree

    def visit_BinOp(self, node):
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == DIV:
            return self.visit(node.left) / self.visit(node.right)
        elif node.op.type == MOD:
            return self.visit(node.left) % self.visit(node.right)
        elif node.op.type == EQUAL:
            return self.visit(node.left) == self.visit(node.right)
        elif node.op.type == NOT:
            return self.visit(node.left) != self.visit(node.right)
        elif node.op.type == GRE:
            return self.visit(node.left) >= self.visit(node.right)
        elif node.op.type == LESE:
            return self.visit(node.left) <= self.visit(node.right)
        elif node.op.type == GRET:
            return self.visit(node.left) > self.visit(node.right)
        elif node.op.type == LEST:
            return self.visit(node.left) < self.visit(node.right)

    def visit_Num(self, node):
        return node.value

    def visit_UnaryOp(self, node):
        op = node.op.type
        if op == PLUS:
            return +self.visit(node.expr)
        elif op == MINUS:
            return -self.visit(node.expr)

    def visit_Compound(self, node):
        for child in node.children:
            self.visit(child)

    def visit_Assign(self, node):
        var_name = node.left.value
        var_value = self.visit(node.right)
        self.GLOBAL_MEMORY[var_name] = var_value

    def visit_Var(self, node):
        var_name = node.value
        var_value = self.GLOBAL_MEMORY.get(var_name)
        if var_value is None:
            raise NameError(repr(var_name))
        return var_value

    def visit_If(self, node):
        if self.visit(node.condition):
            self.visit(node.body)
        else:
            self.visit(node.rest)

    def visit_Else(self, node):
        self.visit(node.body)

    def visit_While(self, node):
        while self.visit(node.condition):
            self.visit(node.body)

    def visit_list(self, node):
        for child in node:
            self.visit(child)

    def visit_NoOp(self, node):
        pass

    def interpret(self):
        tree = self.tree
        if tree is None:
            return ''
        return self.visit(tree)
        






def main():
    file = open("input_test1.txt", "r")
    text = file.read()
    lex = Lexer(text)
    # for i in range(0,15):
    #     print(lex.get_next_token())
    par = Parser(lex)
    tree = par.parse()
    print(tree)
    inter = Interpreter(tree)
    result = inter.interpret()
    print('Run-time GLOBAL_MEMORY contents:')
    print(inter.GLOBAL_MEMORY)

if __name__ == '__main__':
    main()