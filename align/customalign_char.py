#!/usr/bin/env python
'''
Simple Smith-Waterman aligner
'''
import sys

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

        ref = ref.upper()
        query = query.upper()

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

        # backtrack from max
        row = max_row
        col = max_col
        val = max_val

        op = ''
        aln = []

        path = []
        while True:
            val, op = matrix.get(row, col)

            if val <= 0:
                break

            path.append((row, col))
            aln.append(op)

            if op == 'm':
                row -= 1
                col -= 1
            elif op == 'i':
                row -= 1
            elif op == 'd':
                col -= 1
            else:
                break

        aln.reverse()

        cigar = _reduce_cigar(aln)
        return Alignment(orig_query, orig_ref, row, col, cigar, max_val)

    def dump_matrix(self, ref, query, matrix, path, show_row=-1, show_col=-1):
        sys.stdout.write('      -      ')
        sys.stdout.write('       '.join(ref))
        sys.stdout.write('\n')
        for row in range(matrix.rows):
            if row == 0:
                sys.stdout.write('-')
            else:
                sys.stdout.write(query[row - 1])

            for col in range(matrix.cols):
                if show_row == row and show_col == col:
                    sys.stdout.write('       *')
                else:
                    sys.stdout.write(' %5s%s%s' % (matrix.get(row, col)[0], matrix.get(row, col)[1], '$' if (row, col) in path else ' '))
            sys.stdout.write('\n')


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


def _cigar_str(cigar):
    out = ''
    for num, op in cigar:
        out += '%s%s' % (num, op)
    return out


class Alignment(object):
    def __init__(self, query, ref, q_pos, r_pos, cigar, score):
        self.query = query
        self.ref = ref
        self.q_pos = q_pos
        self.r_pos = r_pos
        self.cigar = cigar
        self.score = score

        self.r_offset = 0
        self.r_region = None

        self.orig_query = query
        self.query = query.upper()

        self.orig_ref = ref
        self.ref = ref.upper()
        
        self.result_ref = ""

        q_len = 0
        r_len = 0

        self.matches = 0
        self.mismatches = 0

        i = self.r_pos
        j = self.q_pos

        for count, op in self.cigar:
            if op == 'm':
                q_len += count
                r_len += count
                for _ in range(count):
                    self.result_ref += self.orig_ref[i]
                    if self.query[j] == self.ref[i]:
                        self.matches += 1
                    else:
                        self.mismatches += 1
                    i += 1
                    j += 1

            elif op == 'i':
                q_len += count
                j += count
                self.mismatches += count
            elif op == 'd':
                r_len += count
                self.mismatches += count
                for _ in range(count):
                    self.result_ref += self.orig_ref[i]
                    i += 1

        self.q_end = q_pos + q_len
        self.r_end = r_pos + r_len
        if self.mismatches + self.matches > 0:
            self.accuracy_on_characters = float(self.matches) / (self.mismatches + self.matches)
        else:
            self.accuracy_on_characters = 0

        self.accuracy_on_query = q_len/ len(self.orig_query)
        self.overall_accuracy = self.accuracy_on_characters * self.accuracy_on_query
    
    def cigar_str(self):
        return _cigar_str(self.cigar)

    def dump(self, wrap=None, out=sys.stdout):
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
                    q += self.orig_query[j]
                    r += self.orig_ref[i]
                    if self.query[j] == self.ref[i]:
                        m += '|'
                    else:
                        m += '.'

                    i += 1
                    j += 1
            elif op == 'd':
                rlen += count
                for k in range(count):
                    q += '-'
                    r += self.orig_ref[i]
                    m += ' '
                    i += 1
            elif op == 'i':
                qlen += count
                for k in range(count):
                    q += self.orig_query[j]
                    r += '-'
                    m += ' '
                    j += 1

            elif op == 'n':
                q += '-//-'
                r += '-//-'
                m += '    '

        q_pre = 'Query: %%%ss ' % 5
        r_pre = 'Ref  : %%%ss ' % 5
        m_pre = ' ' * (8 + 5)

        rpos = self.r_pos
        qpos = self.q_pos

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
                    qpos += 1

            out.write(' %s\n' % qpos)
            

            out.write(m_pre)
            out.write(mfragment)
            out.write('\n')
            out.write(r_pre % (rpos + 1))
            out.write(rfragment)
            for base in rfragment:
                if base != '-':
                    rpos += 1
            out.write(' %s\n\n' % (rpos))

        out.write("Score: %s\n" % self.score)
        out.write("Matches: %s (%.1f%%)\n" % (self.matches, self.accuracy_on_characters * 100))
        out.write("Mismatches: %s\n" % (self.mismatches,))
        out.write("CIGAR: %s\n" % self.cigar)