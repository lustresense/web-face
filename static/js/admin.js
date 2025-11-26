// Admin table + sorting dengan panah arah + edit pasien.
(() => {
  const showRowsSelect = document.getElementById('show-rows');
  const pageInfo = document.getElementById('page-info');
  const prevPageBtn = document.getElementById('prev-page');
  const nextPageBtn = document.getElementById('next-page');
  const pageNumber = document.getElementById('page-number');
  const sortHeaders = document.querySelectorAll('.sortable');
  const bodyEl = document.getElementById('tabel-pasien-body');

  // Arrows
  const arrowMap = {
    nik: document.getElementById('arrow-nik'),
    name: document.getElementById('arrow-name'),
    dob: document.getElementById('arrow-dob'),
    address: document.getElementById('arrow-address')
  };

  // Modal edit
  const modalEdit = document.getElementById('modal-edit');
  const formEdit = document.getElementById('form-edit-pasien');
  const editOldNik = document.getElementById('edit-old-nik');
  const editNik = document.getElementById('edit-nik');
  const editNama = document.getElementById('edit-nama');
  const editDob = document.getElementById('edit-dob');
  const editAlamat = document.getElementById('edit-alamat');
  const btnEditCancel = document.getElementById('btn-edit-cancel');

  let state = {
    patients: [],
    sortKey: 'nik',
    sortDir: 'asc',
    rowsPerPage: 10,
    currentPage: 1
  };

  function updateArrows() {
    Object.keys(arrowMap).forEach(k => {
      arrowMap[k].textContent = '';
    });
    const arrow = state.sortDir === 'asc' ? '▲' : '▼';
    arrowMap[state.sortKey].textContent = arrow;
  }

  function sort() {
    const { sortKey, sortDir } = state;
    state.patients.sort((a,b)=>{
      if(a[sortKey] < b[sortKey]) return sortDir==='asc' ? -1 : 1;
      if(a[sortKey] > b[sortKey]) return sortDir==='asc' ? 1 : -1;
      return 0;
    });
    updateArrows();
  }

  function pageData() {
    const start=(state.currentPage-1)*state.rowsPerPage;
    return state.patients.slice(start,start+state.rowsPerPage);
  }

  function render() {
    bodyEl.innerHTML='';
    const total = state.patients.length;
    const totalPages = Math.ceil(total/state.rowsPerPage)||1;
    state.currentPage = Math.min(state.currentPage,totalPages);
    const data = pageData();
    if(data.length===0){
      bodyEl.innerHTML='<tr><td colspan="5" class="py-6 px-4 text-center text-gray-400">Tidak ada data pasien.</td></tr>';
    } else {
      data.forEach(p=>{
        const tr=document.createElement('tr');
        tr.className='border-b border-border hover:bg-[#202428] transition';
        tr.innerHTML=`
          <td class="py-3 px-4 font-mono text-gray-300 text-sm">${p.nik}</td>
          <td class="py-3 px-4 text-gray-200 text-sm">${p.name}</td>
          <td class="py-3 px-4 text-gray-300 text-sm">${p.dob}</td>
          <td class="py-3 px-4 text-gray-300 text-sm">${p.address}</td>
          <td class="py-3 px-4 flex gap-2">
            <button class="px-3 py-1.5 rounded bg-yellow-600 hover:bg-yellow-700 text-white text-xs btn-edit" data-nik="${p.nik}">Edit</button>
            <form method="post" action="/admin/patient/${p.nik}/delete" onsubmit="return confirm('Hapus pasien NIK ${p.nik}?')">
              <button type="submit" class="px-3 py-1.5 rounded bg-red-600 hover:bg-red-700 text-white text-xs">Hapus</button>
            </form>
          </td>
        `;
        bodyEl.appendChild(tr);
      });
    }
    const startRange = total===0?0:(state.currentPage-1)*state.rowsPerPage+1;
    const endRange = Math.min(state.currentPage*state.rowsPerPage,total);
    pageInfo.textContent=`Menampilkan ${startRange}-${endRange} dari ${total}`;
    pageNumber.textContent=`Halaman ${state.currentPage} / ${totalPages}`;
    prevPageBtn.disabled = state.currentPage===1;
    nextPageBtn.disabled = state.currentPage===totalPages;
  }

  async function fetchPatients(){
    try{
      const r=await fetch('/api/patients');
      const d=await r.json();
      if(!d.ok){alert(d.msg||'Gagal memuat pasien');return;}
      state.patients=d.patients||[];
      sort();render();
    }catch(e){alert('Error jaringan: '+e.message);}
  }

  sortHeaders.forEach(h=>{
    h.addEventListener('click',()=>{
      const key=h.dataset.sort;
      if(state.sortKey===key) state.sortDir = state.sortDir==='asc'?'desc':'asc';
      else { state.sortKey=key; state.sortDir='asc'; }
      sortHeaders.forEach(x=>x.classList.remove('active'));
      h.classList.add('active');
      sort(); state.currentPage=1; render();
    });
  });

  showRowsSelect.addEventListener('change',()=>{
    state.rowsPerPage=parseInt(showRowsSelect.value,10);
    state.currentPage=1; render();
  });

  prevPageBtn.addEventListener('click',()=>{
    if(state.currentPage>1){ state.currentPage--; render(); }
  });

  nextPageBtn.addEventListener('click',()=>{
    const totalPages=Math.ceil(state.patients.length/state.rowsPerPage)||1;
    if(state.currentPage<totalPages){ state.currentPage++; render(); }
  });

  bodyEl.addEventListener('click',e=>{
    const btn=e.target;
    if(btn.classList.contains('btn-edit')){
      const nik=btn.dataset.nik;
      const p=state.patients.find(x=>String(x.nik)===String(nik));
      if(!p){alert('Data tidak ditemukan');return;}
      editOldNik.value=p.nik;
      editNik.value=p.nik;
      editNama.value=p.name;
      editDob.value=p.dob;
      editAlamat.value=p.address;
      modalEdit.classList.remove('hidden');
    }
  });

  btnEditCancel.addEventListener('click',()=>modalEdit.classList.add('hidden'));

  formEdit.addEventListener('submit', async e=>{
    e.preventDefault();
    const fd=new FormData();
    fd.append('old_nik', editOldNik.value.trim());
    fd.append('nik', editNik.value.trim());
    // Nama TIDAK dikirim karena tidak bisa diubah
    fd.append('dob', editDob.value.trim());
    fd.append('address', editAlamat.value.trim());
    try{
      const r=await fetch('/admin/patient/update',{method:'POST',body:fd});
      const d=await r.json();
      if(!d.ok){alert(d.msg||'Gagal update');return;}
      alert(d.msg);
      modalEdit.classList.add('hidden');
      await fetchPatients();
    }catch(err){alert('Error jaringan: '+err.message);}
  });

  fetchPatients();
})();