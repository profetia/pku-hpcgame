
subroutine filter_run(x, wgt, ngrid, is, ie, js, je) bind(C, name="filter_run")
    implicit none
    real(8) :: x(2700, 113)
    real(8) :: wgt(1799, 113)
    integer(4) :: ngrid(113)
    integer(4), value :: is, ie, js, je

    integer(4) :: is_f, ie_f, js_f, je_f
    real(8) :: tmp(ie - is + 1)
    integer(4) :: i, j, p, n, hn
    
    is_f = is + 1
    ie_f = ie + 1
    js_f = js + 1
    je_f = je + 1

    do j = js_f, je_f
        n = ngrid(j)
        hn = (n - 1) / 2
        do i = is_f, ie_f
            tmp(i - is_f + 1) = 0
            do p = 1, n
                tmp(i - is_f + 1) = tmp(i - is_f + 1) + wgt(p, j) * x(i - hn + p - 1, j)
            end do
        end do
        do i = is_f, ie_f
            x(i, j) = tmp(i - is_f + 1)
        end do
    end do
end subroutine filter_run
