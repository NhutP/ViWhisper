#!/usr/bin/env python
'''
Simple Smith-Waterman aligner
'''
import sys

def SpecialUpper(self):
    uppercase_list = []
    for original_string in self:
        uppercase_string = original_string.upper()
        uppercase_list.append(uppercase_string)
    return uppercase_list
    
class IdentityScoringMatrix(object):
    def __init__(self, match=1, mismatch=-1):
        self.match = match
        self.mismatch = mismatch

    def score(self, one, two):
        if one == two:
            return self.match
        return self.mismatch

class Matrix(object):
    def __init__(self, rows, cols, init=None):
        self.rows = rows
        self.cols = cols
        self.values = [init, ] * rows * cols

    def get(self, row, col):
        return self.values[(row * self.cols) + col]

    def set(self, row, col, val):
        self.values[(row * self.cols) + col] = val


class LocalAlignment(object):
    def __init__(self, scoring_matrix, gap_penalty=-1):
        self.scoring_matrix = scoring_matrix
        self.gap_penalty = gap_penalty

    def align(self, ref, query):
        orig_ref = ref
        orig_query = query

        ref = SpecialUpper(ref)
        query = SpecialUpper(query)

        matrix = Matrix(len(query) + 1, len(ref) + 1, (0, ' '))
        for row in range(1, matrix.rows):
            matrix.set(row, 0, (0, 'i'))

        for col in range(1, matrix.cols):
            matrix.set(0, col, (0, 'd'))

        max_val = 0
        max_row = 0
        max_col = 0

        # calculate matrix
        for row in range(1, matrix.rows):
            for col in range(1, matrix.cols):
                mm_val = matrix.get(row - 1, col - 1)[0] + self.scoring_matrix.score(query[row - 1], ref[col - 1])

                ins_val = 0
                if matrix.get(row - 1, col)[0] > 0:
                    ins_val = matrix.get(row - 1, col)[0] + self.gap_penalty
                
                del_val = 0
                if matrix.get(row, col - 1)[0] > 0:
                    del_val = matrix.get(row, col - 1)[0] + self.gap_penalty

                cell_val = max(mm_val, del_val, ins_val, 0)

                if cell_val == mm_val:
                    val = (cell_val, 'm')
                elif cell_val == del_val:
                    val = (cell_val, 'd')
                elif cell_val == ins_val:
                    val = (cell_val, 'i')
                else:
                    val = (0, 'x')

                if val[0] > max_val:
                    max_val = val[0]
                    max_row = row
                    max_col = col

                matrix.set(row, col, val)

        row = max_row
        col = max_col
        val = max_val
        op = ''
        aln = []
        count = 0

        while True:
            val, op = matrix.get(row, col)
            if val <= 0:
                break
            aln.append(op)
            count += 1
            if op == 'm':
                row -= 1
                col -= 1
            elif op == 'i':
                row -= 1
            elif op == 'd':
                col -= 1
            else:
                break

        while count < len(orig_query):
            if orig_ref[col] == orig_query[0]:
                break
            count += 1
            col -= 1
            aln.append('d') 
            if col < 0:
                break
            
        aln.reverse()   
        cigar = _reduce_cigar(aln)
        
        return Alignment(orig_query, orig_ref, row, col, cigar, max_val)

def _reduce_cigar(operations):
    count = 1
    last = None
    ret = []
    for op in operations:
        if last and op == last:
            count += 1
        elif last:
            ret.append((count, last))
            count = 1
        last = op

    if last:
        ret.append((count, last))
    return ret

class Alignment(object):
    def __init__(self, query, ref, q_pos, r_pos, cigar, score):
        self.query = query
        self.ref = ref
        self.q_pos = q_pos
        self.r_pos = r_pos
        self.cigar = cigar
        self.score = score
        self.ori_qlen = len(query)       

        self.orig_query = query
        self.query = SpecialUpper(query)
        self.orig_ref = ref
        self.ref = SpecialUpper(ref)
        
        self.result_query = ""
        self.result_ref = ""
        
        q_len = 0
        r_len = 0

        self.matches = 0
        self.mismatches = 0
        self.gap_count = 0
        
        i = self.r_pos
        j = self.q_pos

        q = ''
        r = ''

        for count, op in self.cigar:
            if op == 'm':
                q_len += count
                r_len += count
                for _ in range(count):
                    if self.query[j] == self.ref[i]:
                        self.matches += 1
                    else:
                        self.mismatches += 1
                    q += self.orig_query[j] + ' '
                    r += self.orig_ref[i] + ' '
                    i += 1
                    j += 1

            elif op == 'i':
                q_len += count
                self.gap_count += count
                r += '- ' * count
                for _ in range(count):
                    q += self.orig_query[j] + ' '
                    j += 1
            elif op == 'd':
                r_len += count
                self.gap_count += count
                q += '- ' * count
                for _ in range(count):
                    r += self.orig_ref[i] + ' '
                    i += 1
                
        while q_len < len(self.orig_query) and j < len(self.query) and i < len(self.ref):
            if self.ref[i] == self.query[j]:
                self.matches += 1
            else:
                self.mismatches += 1
            q += self.orig_query[j] + ' '
            r += self.orig_ref[i] + ' '
            i += 1
            j += 1
            q_len += 1
            r_len += 1
            
        self.q_end = q_pos + q_len - 1
        self.r_end = r_pos + r_len - 1
        
        self.result_query = q[:-1]
        self.result_ref = r[:-1]
    
        if self.mismatches + self.matches > 0:
            self.accuracy_on_words = float(self.matches) / (self.mismatches + self.matches + self.gap_count)
        else:
            self.accuracy_on_words = 0

        self.accuracy_on_query = q_len/ len(self.orig_query)
        self.overall_accuracy = self.accuracy_on_words * self.accuracy_on_query
        # print(self.overall_accuracy)

    def _cigar_str(self):
        out = ''
        for num, op in self.cigar:
            out += '%s%s' % (num, op)
        return out
    

    def ImportantInformation(self):
        out = sys.stdout
        out.write('Query: %s %s %s\n' % (self.q_pos, self.result_query, self.q_end))
        out.write('Ref: %s %s %s\n' % (self.r_pos, self.result_ref, self.r_end))
        out.write('Precision: %s \n' % self.precision)
        # return self.result_ref

    def word_dump(self, out=sys.stdout):
        i = self.r_pos
        j = self.q_pos

        q = ''
        m = ''
        r = ''
        qlen = 0
        rlen = 0

        for count, op in self.cigar:
            if op == 'm':
                qlen += count
                rlen += count
                for k in range(count):
                    q += self.orig_query[j] + ' '
                    r += self.orig_ref[i] + ' '
                    if self.query[j] == self.ref[i]:
                        m += '|'
                    else:
                        m += '.'
                    i += 1
                    j += 1
            elif op == 'd':
                rlen += count
                for k in range(count):
                    q += '- '
                    r += self.orig_ref[i] + ' '
                    m += ' '
                    i += 1
            elif op == 'i':
                qlen += count
                for k in range(count):
                    q += self.orig_query[j] + ' '
                    r += '- '
                    m += '  '
                    j += 1

            elif op == 'n':
                q += '-//- '
                r += '-//- '
                m += '    '

        q = q[:-1]
        r = r[:-1]
        self.qlen = qlen
        self.rlen = rlen
        
        
        q_pre = 'Query: %%%ss ' % 5
        r_pre = 'Ref  : %%%ss ' % 5
        m_pre = ' ' * (10)

        rpos = self.r_pos
        qpos = self.q_end
            
        while q and r and m:
            out.write(q_pre % (qpos + 1))  # pos is displayed as 1-based

            qfragment = q
            mfragment = m
            rfragment = r
            
            q = ''
            m = ''
            r = ''

            out.write(qfragment)
            
            for base in qfragment:
                if base != '-':
                    qpos -= 1

            qpos = self.q_pos + qlen
            out.write(' %s\n' % qpos)
            

            out.write(m_pre)
            out.write(mfragment)
            out.write('\n')
            out.write(r_pre % (rpos + 5))
            out.write(rfragment)
            for base in rfragment:
                if base != '-':
                    rpos += 1
            rpos = self.r_pos + rlen
            out.write(' %s\n\n' % (rpos + 5))

        out.write("Score: %s\n" % self.score)
        out.write("Matches: %s (%.1f%%)\n" % (self.matches, self.identity * 100))
        out.write("Mismatches: %s\n" % (self.mismatches,))
        out.write("CIGAR: %s\n" % self._cigar_str())
        out.write("Length: %s\n" % qlen)
        out.write("Original length: %s\n" % self.ori_qlen)
        out.write("Precision on word: %s\n" % (qlen/self.ori_qlen))